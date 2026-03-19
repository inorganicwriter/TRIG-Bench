import os
import argparse
import json
import random
import time
from data_collector.comfy_client import ComfyClient
from data_collector.utils import load_workflow_api

# Key Node IDs (must match image_qwen_image_edit.json)
NODE_ID_LOAD_IMAGE = "78"
NODE_ID_PROMPT = "76"
NODE_ID_KSAMPLER = "3"

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Generator (LLM + ComfyUI)")
    parser.add_argument("--attack-file", type=str, required=True, help="Path to attacks.jsonl (from generate_attacks.py)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save final benchmark images")
    parser.add_argument("--comfy-server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    return parser.parse_args()

def generate_image_with_comfy(client, workflow_template, input_image_path, prompt, seed=None):
    """
    Helper function to run the ComfyUI workflow for a single image generation.
    Returns: image_data (bytes) or None if failed.
    """
    # 1. Upload Input Image
    comfy_filename = client.upload_image(input_image_path)
    if not comfy_filename:
        print(f"    -> Upload failed for {input_image_path}")
        return None, None

    # 2. Configure Workflow
    workflow = workflow_template.copy()
    
    if NODE_ID_LOAD_IMAGE in workflow:
        workflow[NODE_ID_LOAD_IMAGE]["inputs"]["image"] = comfy_filename
    
    if NODE_ID_PROMPT in workflow:
        workflow[NODE_ID_PROMPT]["inputs"]["prompt"] = prompt
        
    if seed is None:
        seed = random.randint(1, 10**14)
    
    if NODE_ID_KSAMPLER in workflow:
        workflow[NODE_ID_KSAMPLER]["inputs"]["seed"] = seed

    # 3. Queue Prompt
    prompt_res = client.queue_prompt(workflow)
    if not prompt_res:
        print("    -> Queue failed.")
        return None, None
    
    prompt_id = prompt_res['prompt_id']
    
    # 4. Wait for Completion
    if not client.wait_for_completion(prompt_id):
        print("    -> Timeout/Error.")
        return None, None
        
    # 5. Retrieve Image
    history = client.get_history(prompt_id)
    if not history: 
        return None, None
    
    history_data = history[prompt_id]
    for node_id in history_data['outputs']:
        node_output = history_data['outputs'][node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = client.get_image(image['filename'], image['subfolder'], image['type'])
                if image_data:
                    return image_data, seed # Return data and used seed
    
    return None, None

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Initialize ComfyUI Client
    print(f"Connecting to ComfyUI at {args.comfy_server}...")
    client = ComfyClient(args.comfy_server)
    if not client.connect():
        print("Failed to connect to ComfyUI. Exiting.")
        return

    # 2. Load Workflow Template
    workflow_path = os.path.join("data_collector", "image_qwen_image_edit.json")
    if not os.path.exists(workflow_path):
        print(f"Error: Workflow file not found at {workflow_path}")
        return
        
    workflow_template = load_workflow_api(workflow_path)
    if not workflow_template:
        print("Error: Failed to load workflow template.")
        return

    # 3. Load Attacks
    print(f"Reading attacks from {args.attack_file}...")
    attacks = []
    with open(args.attack_file, 'r', encoding='utf-8') as f:
        for line in f:
            attacks.append(json.loads(line))
            
    print(f"Found {len(attacks)} entries.")

    # 4. Process Loop
    processed_count = 0
    
    for i, entry in enumerate(attacks):
        if args.limit > 0 and processed_count >= args.limit:
            break
            
        original_filename = entry.get('original_filename')
        base_name = os.path.splitext(original_filename)[0]
        
        # Locate Original Clean Image
        clean_img_path = entry.get('image_path')
        if not clean_img_path or not os.path.exists(clean_img_path):
            # Fallback attempt
            clean_img_path = os.path.join("data", "clean_images", entry.get('clean_image_path', ''))
            if not os.path.exists(clean_img_path):
                print(f"Warning: Clean image not found for {original_filename}. Skipping.")
                continue

        print(f"\n[{i+1}/{len(attacks)}] Processing {original_filename}...")
        
        attack_dict = entry.get('attacks', {})
        text_location = entry.get('text_location', 'in the image')
        
        # ==========================================
        # Resume Support: Skip if all outputs already exist
        # ==========================================
        blank_subdir = "Blank"
        blank_save_dir = os.path.join(args.output_dir, blank_subdir)
        blank_save_name = f"{base_name}_{blank_subdir}.png"
        blank_save_path = os.path.join(blank_save_dir, blank_save_name)
        
        # Check if ALL attack images already exist for this entry
        all_exist = os.path.exists(blank_save_path)
        if all_exist:
            for attack_type, attack_text in attack_dict.items():
                safe_text = "".join([c for c in attack_text if c.isalnum() or c in (' ', '_', '-')]).strip()
                subdir = attack_type.capitalize()
                candidate = f"{base_name}_{attack_type}_{safe_text}.png"
                if len(candidate.encode('utf-8')) > 255:
                    max_text_len = 255 - len(f"{base_name}_{attack_type}_.png".encode('utf-8'))
                    safe_text = safe_text[:max(10, max_text_len)]
                    candidate = f"{base_name}_{attack_type}_{safe_text}.png"
                check_path = os.path.join(args.output_dir, subdir, candidate)
                if not os.path.exists(check_path):
                    all_exist = False
                    break
        
        if all_exist:
            print(f"  > Already complete, skipping.")
            processed_count += 1
            continue
        
        # ==========================================
        # Phase 1: Generate BLANK Image (Control Group)
        # ==========================================
        # Prompt: "Remove the text..."
        blank_prompt = f"Remove the text {text_location}. Maintain photorealism and fill the area naturally."
        print(f"  > [Phase 1] Generating Blank Image (Removing text)...")
        
        blank_image_data, blank_seed = generate_image_with_comfy(
            client, workflow_template, clean_img_path, blank_prompt
        )
        
        if not blank_image_data:
            print(f"    -> Failed to generate Blank image. Skipping attacks for this image.")
            continue
            
        # Save Blank Image
        blank_subdir = "Blank"
        blank_save_dir = os.path.join(args.output_dir, blank_subdir)
        os.makedirs(blank_save_dir, exist_ok=True)
        blank_save_name = f"{base_name}_{blank_subdir}.png"
        blank_save_path = os.path.join(blank_save_dir, blank_save_name)
        
        with open(blank_save_path, 'wb') as f:
            f.write(blank_image_data)
        print(f"    -> Saved Blank: {os.path.join(blank_subdir, blank_save_name)}")
        
        # Save Blank Metadata
        meta_entry = {
            "filename": blank_save_name,
            "original_source": original_filename,
            "source_image_used": clean_img_path, # Blank is derived from Clean
            "injected_text": "",
            "attack_type": "Blank",
            "prompt_used": blank_prompt,
            "seed": blank_seed
        }
        with open(os.path.join(args.output_dir, "benchmark_meta.jsonl"), "a", encoding="utf-8") as meta_f:
            meta_f.write(json.dumps(meta_entry) + "\n")

        # ==========================================
        # Phase 2: Generate Attacks (Using ORIGINAL CLEAN IMAGE as input)
        # ==========================================
        # Reverted strategy: Use Original Image to preserve text geometry/style
        
        for attack_type, attack_text in attack_dict.items():
            if attack_type == 'Blank': 
                continue # Already done
                
            # Construct Prompt: "Replace the text..." to leverage original context
            prompt = f"Replace the text {text_location} with '{attack_text}'. Maintain photorealism and natural appearance."
            
            print(f"  > [Phase 2] Generating {attack_type}: '{attack_text}'")
            
            # Use clean_img_path directly
            attack_image_data, attack_seed = generate_image_with_comfy(
                client, workflow_template, clean_img_path, prompt 
            )
            
            if not attack_image_data:
                print(f"    -> Failed to generate {attack_type}.")
                continue
                
            # Save Attack Image
            subdir = attack_type.capitalize()
            save_dir = os.path.join(args.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            
            # Sanitize filename (remove special chars)
            safe_text = "".join([c for c in attack_text if c.isalnum() or c in (' ', '_', '-')]).strip()
            save_name = f"{base_name}_{attack_type}_{safe_text}.png"
            
            # Only truncate if filename exceeds Linux 255-byte limit
            if len(save_name.encode('utf-8')) > 255:
                max_text_len = 255 - len(f"{base_name}_{attack_type}_.png".encode('utf-8'))
                safe_text = safe_text[:max(10, max_text_len)]
                save_name = f"{base_name}_{attack_type}_{safe_text}.png"
            save_path = os.path.join(save_dir, save_name)
            
            with open(save_path, 'wb') as f:
                f.write(attack_image_data)
            print(f"    -> Saved: {os.path.join(subdir, save_name)}")
            
            # Save Metadata
            meta_entry = {
                "filename": save_name,
                "original_source": original_filename,
                "clean_source": clean_img_path, # Back to Clean Source
                "injected_text": attack_text,
                "attack_type": attack_type,
                "prompt_used": prompt,
                "seed": attack_seed
            }
            with open(os.path.join(args.output_dir, "benchmark_meta.jsonl"), "a", encoding="utf-8") as meta_f:
                meta_f.write(json.dumps(meta_entry) + "\n")

        processed_count += 1

    print(f"\nBenchmark Generation Complete. Metadata saved to {os.path.join(args.output_dir, 'benchmark_meta.jsonl')}")
    client.close()

if __name__ == "__main__":
    main()
