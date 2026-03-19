import os
import argparse
import json
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path to allow running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm.asyncio import tqdm
from data_collector.llm_provider import OpenAICompatibleProvider

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Adversarial Attacks using VLMs")
    parser.add_argument("--clean-meta", type=str, required=True, help="Path to clean_images metadata.jsonl")
    parser.add_argument("--original-dir", type=str, required=True, help="Directory containing original raw images")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for attack configurations")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Thinking", help="VLM Model Name")
    parser.add_argument("--api-base", type=str, default="http://localhost:8001/v1", help="API Base URL")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for vLLM")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    return parser.parse_args()

async def process_single_image(provider, image_path, original_filename, clean_img_rel_path):
    """
    Process a single image to generate attacks using the provider.
    """
    prompt = """
    Analyze the text in this street view image.
    
    Task:
    1. Identify the main text content (e.g., store names, road signs).
    2. Describe WHERE the text is located in the image using natural language (e.g., "on the green storefront sign at the top center", "on the blue street sign on the left").
    3. IF NO LEGIBLE TEXT IS FOUND, do NOT generate any attacks. Return an empty "attacks" object.
    4. If text is found, generate 3 types of short distraction texts to replace it:
       - "Similar": Visually or semantically similar (e.g., McDonald's -> McDonalds).
       - "Random": A completely unrelated word or short phrase.
       - "Adversarial": Text that conveys the opposite meaning or misleading info (e.g., 'Stop' -> 'Go', or a different city name 'Paris').
    
    Output JSON format ONLY:
    {
        "original_text": "...", (or null if no text found)
        "text_location": "...", (natural language description of where the text is in the image)
        "attacks": {
            "similar": "...",
            "random": "...",
            "adversarial": "..."
        }
    }
    """
    
    result = await provider.analyze_image_async(
        image_path=Path(image_path),
        prompt=prompt,
        json_mode=True # Provider handles Thinking models automatically
    )
    
    if result.success and result.content:
        try:
            attack_data = json.loads(result.content)
            attacks = attack_data.get("attacks", {})
            
            # Filter out if no text found / no attacks generated
            if not attacks:
                return None
                
            return {
                "original_filename": original_filename,
                "clean_image_path": clean_img_rel_path, # relative path
                "image_path": image_path,
                "original_text": attack_data.get("original_text", ""),
                "text_location": attack_data.get("text_location", "in the image"),
                "attacks": attacks
            }
        except json.JSONDecodeError:
            # print(f"JSON Parse Error for {original_filename}")
            pass
    
    return None

async def main_async():
    args = parse_args()
    
    # Initialize Provider
    provider = OpenAICompatibleProvider(
        model_name=args.model,
        base_url=args.api_base,
        api_key=args.api_key,
        max_tokens=2048,
        temperature=0.7
    )
    
    if not provider.is_available():
        print("Error: LLM Provider is not available. Check your API connection.")
        return

    # Load Clean Metadata
    print(f"Loading metadata from {args.clean_meta}...")
    clean_entries = []
    with open(args.clean_meta, 'r', encoding='utf-8') as f:
        for line in f:
            clean_entries.append(json.loads(line))
            
    print(f"Found {len(clean_entries)} images.")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process Loop
    tasks = []
    processed_count = 0
    skipped_count = 0
    
    # To avoid overwhelming the server, we might want to semaphore, 
    # but vLLM handles batching well. Let's use a semaphore of 20.
    sem = asyncio.Semaphore(20)
    
    async def sem_task(entry):
        nonlocal skipped_count
        async with sem:
            # Construct original image path
            # Assuming filename in metadata matches file in original-dir
            fname = entry.get('filename') # e.g. London.jpg
            if not fname: 
                skipped_count += 1
                return None
            
            # The clean image output filename is stored in 'output_filename' in clean metadata
            clean_rel_path = entry.get('output_filename')
            
            original_path = os.path.join(args.original_dir, fname)
            if not os.path.exists(original_path):
                # Try finding with extensions
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    if os.path.exists(original_path + ext):
                        original_path = original_path + ext
                        found = True
                        break
                
                if not found:
                    # File not in filtered_images, skip silently
                    skipped_count += 1
                    return None
                
            return await process_single_image(provider, original_path, fname, clean_rel_path)

    results = []
    for i, entry in enumerate(clean_entries):
        if args.limit > 0 and i >= args.limit:
            break
        tasks.append(sem_task(entry))

    print(f"Generating attacks for {len(tasks)} images...")
    
    # Execute with progress bar and incremental save
    completed_tasks = []
    
    # Open file for appending (or overwrite initially if needed, handled by main logic before loop?)
    # For safety, we open in 'w' before loop
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for f in tqdm.as_completed(tasks, total=len(tasks)):
            res = await f
            if res:
                completed_tasks.append(res)
                # Write immediately
                f_out.write(json.dumps(res) + "\n")
                f_out.flush() # Ensure content is written to disk
            
    # Save Results Summary
    print(f"\n--- Summary ---")
    print(f"Total in metadata: {len(clean_entries)}")
    print(f"Files not found (skipped): {skipped_count}")
    print(f"LLM returned empty: {len(clean_entries) - skipped_count - len(completed_tasks)}")
    print(f"Successful attacks: {len(completed_tasks)}")
    print(f"Saved {len(completed_tasks)} attack configurations to {args.output}")
            
    print("Done.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
