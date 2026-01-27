import os
import json
import argparse
import time
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Adversarial Texts using LLM (Qwen-VL)")
    parser.add_argument("--clean-meta", type=str, required=True, help="Path to metadata.jsonl from Step 1 (Text Removal)")
    parser.add_argument("--original-dir", type=str, required=True, help="Directory containing ORIGINAL images (Step 0)")
    parser.add_argument("--output", type=str, required=True, help="Path to save attacks.jsonl")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="LLM API Base URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    return parser.parse_args()

def generate_attacks(client, model, image_path):
    """
    Uses Qwen-VL to: 1. Read text, 2. Generate attacks
    """
    base64_image = encode_image(image_path)
    
    prompt = """
    Analyze the text in this street view image.
    Task:
    1. Identify the main text content (e.g., store names, road signs).
    2. Based on that text, generate 3 types of short distraction texts to replace it:
       - "Similar": Visually or semantically similar (e.g., McDonald's -> McDonalds).
       - "Random": A completely unrelated word or short phrase.
       - "Adversarial": Text that conveys the opposite meaning or misleading info (e.g., 'Stop' -> 'Go').
    
    Output JSON format ONLY:
    {
        "original_text": "...",
        "attacks": {
            "similar": "...",
            "random": "...",
            "adversarial": "..."
        }
    }
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error generating attack for {image_path}: {e}")
        return None

def main():
    args = parse_args()
    
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    
    # Load processed images list
    processed_entries = []
    with open(args.clean_meta, 'r', encoding='utf-8') as f:
        for line in f:
            processed_entries.append(json.loads(line))
            
    print(f"Found {len(processed_entries)} entries in metadata.")
    
    results = []
    
    for i, entry in enumerate(processed_entries):
        # We need the ORIGINAL image to read text.
        # Step 1 metadata should have 'original_filename'.
        original_name = entry.get('original_filename')
        if not original_name: continue
        
        orig_path = os.path.join(args.original_dir, original_name)
        if not os.path.exists(orig_path):
            print(f"Warning: Original image not found at {orig_path}")
            continue
            
        print(f"[{i+1}/{len(processed_entries)}] Analyzing {original_name}...")
        
        attack_data = generate_attacks(client, args.model, orig_path)
        
        if attack_data:
            # Combine info for Step 3
            result_entry = {
                "original_filename": original_name,
                "clean_image_path": entry.get('output_filename'), # This is the input for Step 3
                "image_path": os.path.join(os.path.dirname(args.clean_meta), entry.get('output_filename')), # Absolute path estimate
                "detected_text": attack_data.get("original_text", ""),
                "attacks": attack_data.get("attacks", {})
            }
            results.append(result_entry)
            
            # Incremental save
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_entry) + "\n")
                
    print(f"Done. Generated {len(results)} attack configurations to {args.output}")

if __name__ == "__main__":
    main()
