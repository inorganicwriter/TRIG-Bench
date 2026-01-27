import argparse
import os
import json
import base64
from urllib.parse import urlparse
from evaluation.vllm_client import VLLMInferenceClient
from evaluation.metric_calculator import MetricCalculator

# 默认配置
DEFAULT_API_BASE = "http://localhost:8001/v1"
DEFAULT_API_KEY = "qwen-local-key"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Geolocation Robustness")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory containing images to evaluate")
    parser.add_argument("--metadata-file", type=str, required=True, help="Path to YFCC4k metadata text file")
    parser.add_argument("--bench-meta", type=str, required=False, help="Path to benchmark_meta.jsonl (for TFR/TBS pairing)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for API")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="vLLM API Base URL")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API Key")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images")
    return parser.parse_args()

def load_ground_truth(metadata_path):
    gt_map = {}
    print(f"Loading ground truth from {metadata_path}...")
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 15: continue
                try:
                    lon = float(parts[10])
                    lat = float(parts[11])
                    url = parts[14]
                    filename = os.path.basename(urlparse(url).path)
                    gt_map[filename] = (lat, lon)
                    gt_map[parts[1]] = (lat, lon)
                except ValueError: continue
    except Exception as e:
        print(f"Error reading metadata: {e}")
    print(f"Loaded {len(gt_map)} ground truth entries.")
    return gt_map

def load_benchmark_meta(meta_path):
    """
    Load benchmark meta to map filenames to original source (for TBS) and trap info.
    """
    if not meta_path or not os.path.exists(meta_path):
        return {}
    
    meta_map = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                meta_map[entry['filename']] = entry
            except: pass
    return meta_map

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    args = parse_args()
    
    gt_map = load_ground_truth(args.metadata_file)
    bench_meta = load_benchmark_meta(args.bench_meta)
    
    if not gt_map: return

    # Init Client
    client = VLLMInferenceClient(args.api_base, args.api_key, args.model)
    
    # Scan images
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images in {args.img_dir}")
    
    processed_count = 0
    results_buffer = []

    # Store clean results for TBS calculation: {original_base_name: error_km}
    clean_results_map = {} 

    # First Pass: Inference
    for filename in image_files:
        if args.limit > 0 and processed_count >= args.limit: break
            
        gt = gt_map.get(filename)
        # Try finding GT via base name if file is modified
        # E.g. London_similar_Londom.png -> London.jpg
        meta_info = bench_meta.get(filename)
        original_source = meta_info.get('original_source') if meta_info else None
        
        if not gt and original_source:
             gt = gt_map.get(original_source)
        
        if not gt:
             # Try simple split heuristic
             base_id = filename.split('_')[0]
             gt = gt_map.get(base_id)
        
        if not gt: continue

        processed_count += 1
        image_path = os.path.join(args.img_dir, filename)
        
        print(f"[{processed_count}] Evaluating {filename}...")
        
        # Inference
        base64_img = encode_image(image_path)
        pred_text = client.predict_location(base64_img)
        pred_lat, pred_lon = client.parse_coordinates(pred_text)
        
        # Metric: Error Distance
        error_km = None
        if pred_lat is not None:
            error_km = MetricCalculator.haversine_distance(gt[0], gt[1], pred_lat, pred_lon)

        # Metric: WLA
        wla_score = MetricCalculator.calculate_wla(error_km)

        # Cache for TBS (if this is a clean image)
        # We assume 'clean' type or if filename has no attack keywords? 
        # Better to rely on metadata 'attack_type'
        attack_type = meta_info.get('attack_type', 'unknown') if meta_info else 'unknown'
        if attack_type == 'clean' and original_source and error_km is not None:
            clean_results_map[original_source] = error_km
        
        # Prepare Result Entry
        res = {
            "filename": filename,
            "original_source": original_source,
            "attack_type": attack_type,
            "injected_text": meta_info.get('injected_text') if meta_info else None,
            "prediction_text": pred_text,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "gt_lat": gt[0],
            "gt_lon": gt[1],
            "error_km": error_km,
            "wla_score": wla_score
        }
        results_buffer.append(res)
        
        if error_km is not None:
             print(f"  -> Error: {error_km:.2f} km | WLA: {wla_score:.1f}")
        else:
             print(f"  -> Failed to parse: {pred_text}")

    # Second Pass: Calculate TBS & Summary
    # We need to know the trap location for TFR.
    # Limitation: We don't have trap Lat/Lon in bench_meta usually.
    # Placeholder: We check if injected_text matches prediction text? 
    # Or strict TFR requires geocoding. Detailed evaluation might require a separate geocoding step.
    
    total_wla = 0
    valid_count = 0
    tbs_sum = 0
    tbs_count = 0
    
    final_output_path = args.output
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
        
    for res in results_buffer:
        # Calculate TBS
        tbs = None
        if res['attack_type'] != 'clean' and res['original_source'] in clean_results_map:
            clean_err = clean_results_map[res['original_source']]
            adv_err = res['error_km']
            if adv_err is not None:
                tbs = MetricCalculator.calculate_tbs(clean_err, adv_err)
                tbs_sum += tbs
                tbs_count += 1
        res['tbs'] = tbs
        
        if res['error_km'] is not None:
            total_wla += res['wla_score']
            valid_count += 1
        
        # Write Result
        with open(final_output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res) + "\n")

    print("\n" + "="*30)
    print("  TRIG-Bench Evaluation Report")
    print("="*30)
    if valid_count > 0:
        print(f"Mean WLA Score: {total_wla/valid_count:.4f}")
    if tbs_count > 0:
        print(f"Mean TBS Score: {tbs_sum/tbs_count:.2f} km")
    else:
        print("Mean TBS Score: N/A (No paired clean/adv samples found)")
        
    print(f"Detailed results saved to {final_output_path}")

if __name__ == "__main__":
    main()
