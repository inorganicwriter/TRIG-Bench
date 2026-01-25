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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    args = parse_args()
    
    gt_map = load_ground_truth(args.metadata_file)
    if not gt_map: return

    # Init Client
    client = VLLMInferenceClient(args.api_base, args.api_key, args.model)
    
    # Scan images
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images in {args.img_dir}")
    
    processed_count = 0
    total_error = 0
    valid_preds = 0
    
    for filename in image_files:
        if args.limit > 0 and processed_count >= args.limit: break
            
        gt = gt_map.get(filename)
        if not gt:
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
        
        # Metric
        error_km = None
        if pred_lat is not None:
            error_km = MetricCalculator.haversine_distance(gt[0], gt[1], pred_lat, pred_lon)

        res = {
            "filename": filename,
            "prediction_text": pred_text,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "gt_lat": gt[0],
            "gt_lon": gt[1],
            "error_km": error_km
        }
        
        if error_km is not None:
            print(f"  -> Error: {error_km:.2f} km")
            total_error += error_km
            valid_preds += 1
        else:
            print(f"  -> Failed to parse: {pred_text}")
            
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res) + "\n")
            
    if valid_preds > 0:
        print(f"\nAvg Error: {total_error / valid_preds:.2f} km")

if __name__ == "__main__":
    main()
