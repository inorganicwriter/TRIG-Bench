import argparse
import os
import json
import base64
from urllib.parse import urlparse
from evaluation.api_client import build_client, GeoLocalizationClient
from evaluation.metric_calculator import MetricCalculator

# 默认配置
DEFAULT_API_BASE = "http://localhost:8001/v1"
DEFAULT_API_KEY = "local-key"
DEFAULT_MODEL = "qwen3-30b"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Geolocation Robustness")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory containing images to evaluate")
    parser.add_argument("--metadata-file", type=str, required=True, help="Path to metadata TSV file")
    parser.add_argument("--bench-meta", type=str, required=False, help="Path to benchmark_meta.jsonl (for TFR/TBS pairing)")
    parser.add_argument("--baseline", type=str, required=False, help="Path to original image results JSONL (for TBS computation)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model short name (e.g., qwen3-30b, gpt-4o) or full path for local vLLM")
    parser.add_argument("--api-base", type=str, default=None,
                        help="Override API base URL (optional, auto-detected from model registry)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (optional, can also use env vars: OPENROUTER_API_KEY, SILICONFLOW_API_KEY, etc.)")
    parser.add_argument("--provider", type=str, default=None,
                        help="Override provider (local/relay/siliconflow/openrouter/openai)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit)")
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
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except PermissionError:
        print(f"  [SKIP] Permission denied: {image_path}")
        return None

def load_invalid_ids(script_dir=None):
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    invalid_file = os.path.join(script_dir, 'invalid_ids.json')
    invalid_ids = set()
    if os.path.exists(invalid_file):
        with open(invalid_file, 'r') as f:
            data = json.load(f)
        for ids in data.values():
            invalid_ids.update(ids)
        print(f"Loaded {len(invalid_ids)} invalid IDs from {invalid_file} (will be skipped during inference)")
    return invalid_ids


def main():
    args = parse_args()

    gt_map = load_ground_truth(args.metadata_file)
    bench_meta = load_benchmark_meta(args.bench_meta)
    invalid_ids = load_invalid_ids()

    if not gt_map:
        print(f"[ERROR] Ground truth map is empty. Please check --metadata-file: {args.metadata_file}")
        return

    # Build unified client
    try:
        client = build_client(
            model_short_name=args.model,
            provider=args.provider,
            api_key=args.api_key,
            api_base=args.api_base,
        )
        print(f"[Client] Model: {client.model_name}")
        print(f"[Client] Provider: {client.provider}")
        print(f"[Client] API Base: {client.api_base}")
        print(f"[Client] Thinking: {client.is_thinking_model}")
    except ValueError as e:
        print(f"[ERROR] Failed to build client: {e}")
        return

    # Scan images
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} images in {args.img_dir}")

    processed_count = 0
    results_buffer = []

    # Store clean results for TBS calculation
    clean_results_map = {}

    # Load baseline (original image results) if provided
    if args.baseline and os.path.exists(args.baseline):
        print(f"Loading baseline from {args.baseline}...")
        with open(args.baseline, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('error_km') is not None:
                        clean_results_map[entry['filename']] = entry['error_km']
                except: pass
        print(f"Loaded {len(clean_results_map)} baseline entries.")

    final_output_path = args.output

    # Load already-processed filenames to support resume
    already_done = set()
    if os.path.exists(final_output_path):
        with open(final_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_done.add(entry['filename'])
                    if entry.get('attack_type') in ('original', 'clean') and entry.get('error_km') is not None:
                        clean_results_map[entry['filename']] = entry['error_km']
                except: pass
        if already_done:
            print(f"Resuming: {len(already_done)} images already processed, skipping.")

    # First Pass: Inference
    for filename in image_files:
        if args.limit > 0 and processed_count >= args.limit: break

        # Resume: skip already-processed files
        if filename in already_done:
            continue

        gt = gt_map.get(filename)
        meta_info = bench_meta.get(filename)
        original_source = meta_info.get('original_source') if meta_info else None

        if not gt and original_source:
            gt = gt_map.get(original_source)
        if not gt:
            name_no_ext = os.path.splitext(filename)[0]
            gt = gt_map.get(name_no_ext)
        if not gt:
            base_id = filename.split('_')[0]
            gt = gt_map.get(base_id)
            if not gt:
                gt = gt_map.get(os.path.splitext(base_id)[0])
        if not gt: continue

        # Skip invalid samples
        base_id = os.path.splitext(filename.split('_')[0])[0]
        if base_id in invalid_ids:
            print(f"  [SKIP] {filename} (invalid sample ID: {base_id})")
            continue

        processed_count += 1
        image_path = os.path.join(args.img_dir, filename)

        print(f"[{processed_count}] Evaluating {filename}...")

        # Encode image
        base64_img = encode_image(image_path)
        if base64_img is None:
            continue

        # Inference
        pred_text = client.predict_location(base64_img)
        pred_lat, pred_lon = GeoLocalizationClient.parse_coordinates(pred_text) if pred_text else (None, None)

        # Metrics
        error_km = None
        if pred_lat is not None:
            error_km = MetricCalculator.haversine_distance(gt[0], gt[1], pred_lat, pred_lon)
        wla_score = MetricCalculator.calculate_wla(error_km)

        attack_type = meta_info.get('attack_type', 'unknown') if meta_info else 'original'
        if attack_type == 'original' and error_km is not None:
            clean_results_map[filename] = error_km

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
            "wla_score": wla_score,
        }
        results_buffer.append(res)

        # Write result immediately (real-time, supports resume)
        with open(final_output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res) + "\n")

        if error_km is not None:
            print(f"  -> Error: {error_km:.2f} km | WLA: {wla_score:.1f}")
        else:
            print(f"  -> Failed to parse: {pred_text}")

    # Second Pass: Calculate TBS & Summary
    total_wla = 0
    valid_count = 0
    tbs_sum = 0
    tbs_count = 0

    for res in results_buffer:
        tbs = None
        if res['attack_type'] not in ('original', 'clean', 'unknown'):
            orig_src = res.get('original_source', '') or ''
            clean_err = clean_results_map.get(orig_src)
            if clean_err is None:
                base_id = orig_src.split('.')[0] if orig_src else ''
                for key in clean_results_map:
                    if key.startswith(base_id):
                        clean_err = clean_results_map[key]
                        break
            if clean_err is not None and res['error_km'] is not None:
                tbs = MetricCalculator.calculate_tbs(clean_err, res['error_km'])
                tbs_sum += tbs
                tbs_count += 1
        res['tbs'] = tbs

        if res['error_km'] is not None:
            total_wla += res['wla_score']
            valid_count += 1

    # Include resumed results in summary
    if already_done:
        with open(final_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry['filename'] in already_done:
                        if entry.get('error_km') is not None:
                            total_wla += entry.get('wla_score', 0)
                            valid_count += 1
                        if entry.get('tbs') is not None:
                            tbs_sum += entry['tbs']
                            tbs_count += 1
                except: pass

    print("\n" + "="*30)
    print("  SIGNPOST-Bench Evaluation Report")
    print("="*30)
    if valid_count > 0:
        print(f"Mean WLA Score: {total_wla/valid_count*100:.2f}%")
    if tbs_count > 0:
        print(f"Mean TBS Score: {tbs_sum/tbs_count:.2f} km")
    else:
        print("Mean TBS Score: N/A (No paired clean/adv samples found)")
    print(f"Detailed results saved to {final_output_path}")

if __name__ == "__main__":
    main()
