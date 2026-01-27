import os
import random
import argparse
import sys
import json
from datetime import datetime
import traceback
from data_collector.comfy_client import ComfyClient
from data_collector.utils import save_metadata, load_workflow_api

# 默认配置
DEFAULT_SERVER = "127.0.0.1:8188"
DEFAULT_WORKFLOW = os.path.join(os.path.dirname(__file__), "image_qwen_image_edit.json")

# 关键节点 ID
NODE_ID_LOAD_IMAGE = "78"
NODE_ID_PROMPT = "76"
NODE_ID_KSAMPLER = "3"

def parse_args():
    parser = argparse.ArgumentParser(description="ComfyUI Batch Image Processor")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing images OR path to attacks.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output folder for results")
    parser.add_argument("--workflow", type=str, default=DEFAULT_WORKFLOW, help="Path to ComfyUI workflow API JSON")
    parser.add_argument("--prompt", type=str, default="Remove all text from the image.", help="Default prompt (used if input is folder)")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, help="ComfyUI server address (ip:port)")
    parser.add_argument("--seed", type=int, help="Fixed seed for reproducibility (optional)")
    parser.add_argument("--mode", type=str, choices=["remove", "inpaint", "custom"], default="custom", help="Processing mode")
    return parser.parse_args()

def run_workflow(filename, file_path, prompt, seed, mode, args, workflow_template, client):
    """Execution logic for a single prompt/image pair"""
    # 1. Upload
    comfy_filename = client.upload_image(file_path)
    if not comfy_filename:
        return None

    # 2. Configure Workflow
    workflow = workflow_template.copy()
    
    if NODE_ID_LOAD_IMAGE in workflow:
        workflow[NODE_ID_LOAD_IMAGE]["inputs"]["image"] = comfy_filename
    
    if NODE_ID_PROMPT in workflow:
        workflow[NODE_ID_PROMPT]["inputs"]["prompt"] = prompt
    
    if NODE_ID_KSAMPLER in workflow:
        workflow[NODE_ID_KSAMPLER]["inputs"]["seed"] = seed
    else:
        # Some workflows might not have KSampler exposed or use different ID
        pass

    # 3. Queue & Wait
    prompt_res = client.queue_prompt(workflow)
    if not prompt_res: return None
    prompt_id = prompt_res['prompt_id']

    if not client.wait_for_completion(prompt_id):
        return None

    # 4. Retrieve
    history = client.get_history(prompt_id)
    if not history: return None
    
    history_data = history[prompt_id]
    outputs = []
    for node_id in history_data['outputs']:
        node_output = history_data['outputs'][node_id]
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = client.get_image(image['filename'], image['subfolder'], image['type'])
                if not image_data: continue

                # Save
                name_part, ext = os.path.splitext(os.path.basename(filename))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{name_part}_{mode}_{seed}.png"
                save_path = os.path.join(args.output, save_name)
                
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                print(f"   -> Saved: {save_name}")
                
                outputs.append({
                    "original_filename": filename,
                    "output_filename": save_name,
                    "seed": seed,
                    "prompt": prompt,
                    "timestamp": timestamp,
                    "mode": mode
                })
    return outputs

def main():
    args = parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    workflow_template = load_workflow_api(args.workflow)
    if not workflow_template: return

    client = ComfyClient(args.server)
    if not client.connect(): return

    # Determine Work Items
    # Items: list of dicts { 'filename': str, 'path': str, 'prompts': [{'type': str, 'text': str}] }
    work_items = []
    
    if args.input.endswith(".jsonl") and os.path.isfile(args.input):
        print(f"Reading tasks from JSONL: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Support both standard metadata logs and attacks.jsonl format
                    img_path = entry.get('image_path') or entry.get('output_filename') # output_filename from Step 1 can be input here
                    if not img_path: continue
                    
                    item = {
                        'filename': os.path.basename(img_path),
                        'path': img_path,
                        'prompts': []
                    }
                    
                    # If it has "attacks" dict (from generate_attacks.py)
                    if 'attacks' in entry and isinstance(entry['attacks'], dict):
                        for attack_type, attack_text in entry['attacks'].items():
                            # Improved prompt for text injection
                            prompt_tmpl = f"A photo of a street scene. The text '{attack_text}' is written on a sign or building naturally. Maintain photorealism."
                            item['prompts'].append({'type': attack_type, 'text': prompt_tmpl}) # Basic instruction, user can refine
                    else:
                        # Fallback for simple metadata file -> use CLI prompt
                        item['prompts'].append({'type': args.mode, 'text': args.prompt})
                        
                    work_items.append(item)
                except Exception as e:
                    print(f"Skipping bad line: {e}")
    
    elif os.path.isdir(args.input):
        image_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [f for f in os.listdir(args.input) if f.lower().endswith(image_extensions)]
        print(f"Reading images from directory: {args.input}")
        for f in files:
            work_items.append({
                'filename': f,
                'path': os.path.join(args.input, f),
                'prompts': [{'type': args.mode, 'text': args.prompt}]
            })
    else:
        print("Error: Input must be a directory or a .jsonl file")
        return

    print(f"Found {len(work_items)} items to process.")

    success_count = 0
    for i, item in enumerate(work_items):
        print(f"\n[{i+1}/{len(work_items)}] Processing: {item['filename']}")
        
        # Determine strict Prompt or iterate list
        for prompt_data in item['prompts']:
            p_text = prompt_data['text']
            p_type = prompt_data['type']
            
            # Seed strategy: fixed if provided, else random
            seed = args.seed if args.seed is not None else random.randint(1, 10**14)
            
            print(f"  > Mode: {p_type} | Prompt: {p_text[:60]}...")
            try:
                results = run_workflow(
                    item['filename'], 
                    item['path'], 
                    p_text, 
                    seed, 
                    p_type, # Use attack type as mode (similar/random/adversarial)
                    args, 
                    workflow_template, 
                    client
                )
                if results:
                    success_count += 1
                    for res in results:
                        save_metadata(args.output, res)
            except KeyboardInterrupt:
                print("\nStopped.")
                sys.exit(0)
            except Exception as e:
                print(f"Error: {e}")

    client.close()
    print(f"\nBatch processing complete.")

if __name__ == "__main__":
    main()