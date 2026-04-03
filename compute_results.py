import json
import os
import sys
import argparse
import numpy as np
from evaluation.metric_calculator import MetricCalculator


def calculate_wla(error_km):
    return MetricCalculator.calculate_wla(error_km)


def get_base_id(filename):
    # e.g. "1114595220_similar_HunTed.png" -> "1114595220"
    # e.g. "482249918.jpg" -> "482249918"
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    return base.split('_')[0]


def analyze_results(res_dir):
    attacks = ['Original', 'Similar', 'Random', 'Adversarial']
    out_data = {}
    
    if not os.path.exists(res_dir):
        return out_data

    # Auto-detect models by scanning filenames: "results_Original_{model}.jsonl"
    models = set()
    for fname in os.listdir(res_dir):
        if fname.startswith("results_Original_") and fname.endswith(".jsonl"):
            m = fname.replace("results_Original_", "").replace(".jsonl", "")
            models.add(m)
    models = list(models)
    
    # Load invalid IDs if exists
    invalid_ids = set()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    invalid_file = os.path.join(base_dir, 'invalid_ids.json')
    if os.path.exists(invalid_file):
        with open(invalid_file, 'r') as f:
            data = json.load(f)
            # Find the matching dataset key based on res_dir
            if 'im2gps' in res_dir.lower(): ds_key = 'IM2GPS3K'
            elif 'yfcc' in res_dir.lower(): ds_key = 'YFCC4K'
            elif 'google' in res_dir.lower(): ds_key = 'GoogleSV'
            else: ds_key = None
            
            if ds_key:
                invalid_ids = set(data.get(ds_key, []))
            
    for m in models:
        out_data[m] = {}
        original_errors = {}
        
        # 1. Load Original Errors mapping by Base ID
        orig_path = os.path.join(res_dir, f'results_Original_{m}.jsonl')
        if os.path.exists(orig_path):
            with open(orig_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        err = entry.get('error_km')
                        if err is not None:
                            original_source = entry.get('original_source')
                            if original_source:
                                base_id = get_base_id(original_source)
                            else:
                                base_id = get_base_id(entry['filename'])
                            if base_id not in invalid_ids:
                                original_errors[base_id] = err
                    except: pass
                    
        # 2. Compute for all attacks
        for a in attacks:
            fpath = os.path.join(res_dir, f'results_{a}_{m}.jsonl')
            if not os.path.exists(fpath): continue
            
            errors = []
            tbs_list = []
            
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        err = entry.get('error_km')
                        if err is not None:
                            errors.append(err)
                            
                            # Calculate TBS
                            original_source = entry.get('original_source')
                            if original_source:
                                base_id = get_base_id(original_source)
                            else:
                                base_id = get_base_id(entry['filename'])
                                
                            if base_id in invalid_ids: continue
                                
                            orig_err = original_errors.get(base_id)
                            if orig_err is not None:
                                tbs_list.append(err - orig_err)
                    except: pass
            
            if errors:
                wla = sum([calculate_wla(e) for e in errors]) / len(errors) * 100
                med_err = np.median(errors)
                mean_tbs = np.mean(tbs_list) if tbs_list else 0.0
                out_data[m][a] = {
                    'WLA': round(wla, 2), 
                    'MedErr': round(float(med_err), 2), 
                    'TBS': round(float(mean_tbs), 2), 
                    'Count': len(errors)
                }
    return out_data


def main():
    parser = argparse.ArgumentParser(description="Compute SIGNPOST-Bench Evaluation Results Summary")
    parser.add_argument("--base-dir", type=str, default="/home/nas/lsr/Data/SIGNPOST-Bench",
                        help="Base directory containing dataset result folders (default: /home/nas/lsr/Data/SIGNPOST-Bench)")
    parser.add_argument("--datasets", nargs='+', default=['im2gps3k', 'yfcc4k', 'googlesv'],
                        help="Datasets to compute results for (default: im2gps3k yfcc4k googlesv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: parsed_results.json in script directory)")
    args = parser.parse_args()

    final_out = {}
    for ds in args.datasets:
        ds_display = ds.upper().replace('4K', '4K').replace('3K', '3K')
        # Normalize display name
        if 'im2gps' in ds.lower(): ds_display = 'IM2GPS3K'
        elif 'yfcc' in ds.lower(): ds_display = 'YFCC4K'
        elif 'google' in ds.lower(): ds_display = 'GoogleSV'
        else: ds_display = ds.upper()
        
        final_out[ds_display] = analyze_results(os.path.join(args.base_dir, ds, 'results'))

    # Save the output file
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parsed_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_out, f, indent=2)

    print("="*60)
    print("  SIGNPOST-Bench Evaluation Results Summary")
    print("="*60)
    for ds, ds_data in final_out.items():
        if not ds_data: continue
        print(f"\n[{ds}]")
        print(f"{'Model':<15} {'Attack':<15} {'WLA (%)':<10} {'MedErr (km)':<15} {'TBS (km)':<15} {'Count'}")
        print("-" * 80)
        for model, model_data in ds_data.items():
            for attack, metrics in model_data.items():
                print(f"{model:<15} {attack:<15} {metrics['WLA']:<10.2f} {metrics['MedErr']:<15.2f} {metrics['TBS']:<15.2f} {metrics['Count']}")
        print("-" * 80)
    print(f"\nMetrics computation complete. JSON saved to {out_path}")


if __name__ == "__main__":
    main()
