import os
import argparse
import json
import random
import time
from data_collector.comfy_client import ComfyClient
from data_collector.utils import load_workflow_api

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Generator (LLM + ComfyUI)")
    parser.add_argument("--attack-file", type=str, required=True, help="Path to attacks.jsonl (from generate_attacks.py)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save final benchmark images")
    parser.add_argument("--comfy-server", type=str, default="127.0.0.1:8188", help="ComfyUI server address")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    return parser.parse_args()

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
    # We reuse the image edit workflow but change the prompt for addition instead of removal
    workflow_path = os.path.join("data_collector", "image_qwen_image_edit.json")
    if not os.path.exists(workflow_path):
        print(f"Error: Workflow file not found at {workflow_path}")
        return
        
    workflow_template = load_workflow_api(workflow_path)
    if not workflow_template:
        print("Error: Failed to load workflow template.")
        return
        
    # Key Node IDs (must match image_qwen_image_edit.json)
    NODE_ID_LOAD_IMAGE = "78"
    NODE_ID_PROMPT = "76"
    NODE_ID_KSAMPLER = "3"

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
        # We need the CLEAN image path (the result of Step 1)
        # generated_attacks.py gives us 'clean_image_path' relative to execution root or absolute?
        # Let's check how generate_attacks.py saves it.
        # It saves: "clean_image_path": entry.get('output_filename')
        # We assume clean images are in "data/clean_images/" usually.
        # But wait, generate_attacks stores the FULL path in 'image_path' key if possible.
        # Let's rely on 'image_path' from attacks.jsonl which is the clean image path.
        
        clean_img_path = entry.get('image_path')
        if not clean_img_path or not os.path.exists(clean_img_path):
            print(f"Warning: Clean image not found for {original_filename} at {clean_img_path}")
            # Try constructing it if missing?
            # Assuming 'data/clean_images' + output_filename
            clean_img_path = os.path.join("data", "clean_images", entry.get('clean_image_path', ''))
            if not os.path.exists(clean_img_path):
                print(f"  -> Still not found at {clean_img_path}. Skipping.")
                continue

        print(f"\n[{i+1}/{len(attacks)}] Processing {original_filename}...")
        
        attack_dict = entry.get('attacks', {})
        
        for attack_type, attack_text in attack_dict.items():
            # Construct Prompt
            # "Add the text '{text}' naturally into the scene."
            prompt = f"Add the text '{attack_text}' to the image naturally. Maintain photorealism."
            
            print(f"  > Generating {attack_type}: '{attack_text}'")
            
            # Prepare Workflow
            # 1. Upload Clean Image
            comfy_filename = client.upload_image(clean_img_path)
            if not comfy_filename:
                print("    -> Upload failed.")
                continue
                
            # 2. Configure
            workflow = workflow_template.copy()
            
            if NODE_ID_LOAD_IMAGE in workflow:
                workflow[NODE_ID_LOAD_IMAGE]["inputs"]["image"] = comfy_filename
            
            if NODE_ID_PROMPT in workflow:
                workflow[NODE_ID_PROMPT]["inputs"]["prompt"] = prompt
                
            seed = random.randint(1, 10**14)
            if NODE_ID_KSAMPLER in workflow:
                workflow[NODE_ID_KSAMPLER]["inputs"]["seed"] = seed
                
            # 3. Queue
            prompt_res = client.queue_prompt(workflow)
            if not prompt_res:
                print("    -> Queue failed.")
                continue
            
            prompt_id = prompt_res['prompt_id']
            
            # 4. Wait
            if not client.wait_for_completion(prompt_id):
                print("    -> Timeout/Error.")
                continue
                
            # 5. Retrieve
            history = client.get_history(prompt_id)
            if not history: continue
            
            history_data = history[prompt_id]
            for node_id in history_data['outputs']:
                node_output = history_data['outputs'][node_id]
                if 'images' in node_output:
                    for image in node_output['images']:
                        image_data = client.get_image(image['filename'], image['subfolder'], image['type'])
                        if not image_data: continue

                        # Save Image
                        base_name = os.path.splitext(original_filename)[0]
                        save_name = f"{base_name}_{attack_type}_{attack_text}.png"
                        save_path = os.path.join(args.output_dir, save_name)
                        
                        with open(save_path, 'wb') as f:
                            f.write(image_data)
                        print(f"    -> Saved: {save_name}")
                        
                        # Save Metadata
                        meta_entry = {
                            "filename": save_name,
                            "original_source": original_filename,
                            "clean_source": clean_img_path,
                            "injected_text": attack_text,
                            "attack_type": attack_type, # similar, random, adversarial
                            "prompt_used": prompt,
                            "seed": seed
                        }
                        
                        meta_path = os.path.join(args.output_dir, "benchmark_meta.jsonl")
                        with open(meta_path, "a", encoding="utf-8") as meta_f:
                            meta_f.write(json.dumps(meta_entry) + "\n")

        processed_count += 1

    print(f"\nBenchmark Generation Complete. Metadata saved to {os.path.join(args.output_dir, 'benchmark_meta.jsonl')}")
    client.close()

if __name__ == "__main__":
    main()
