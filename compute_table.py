"""
compute_table.py - 从 paper/ 目录下的结果文件计算 Table 1 数据
"""
import json, os, numpy as np

def wla(d):
    if d is None: return 0
    return sum(0.2 for t in [1, 25, 200, 750, 2500] if d < t)

all_results = {}
for ds_name, ds_dir in [('IM2GPS3K', 'paper/im2gps3kresults'), ('YFCC4K', 'paper/yfcc4kresults')]:
    all_results[ds_name] = {}
    for model in ['qwen3-30b', 'qwen3-8b']:
        all_results[ds_name][model] = {}
        orig_errors = {}
        orig_file = f'{ds_dir}/results_Original_{model}.jsonl'
        if os.path.exists(orig_file):
            with open(orig_file, encoding='utf-8') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        if e.get('error_km') is not None:
                            base = os.path.splitext(e['filename'])[0]
                            orig_errors[base] = e['error_km']
                    except: pass

        for attack in ['Original', 'Similar', 'Random', 'Adversarial']:
            fpath = f'{ds_dir}/results_{attack}_{model}.jsonl'
            if not os.path.exists(fpath): continue
            errors, tbs_list = [], []
            with open(fpath, encoding='utf-8') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        err = e.get('error_km')
                        if err is not None:
                            errors.append(err)
                            orig_src = str(e.get('original_source', ''))
                            orig_err = orig_errors.get(orig_src) or orig_errors.get(os.path.splitext(orig_src)[0])
                            if orig_err is not None:
                                tbs_list.append(err - orig_err)
                    except: pass
            if errors:
                w = sum(wla(e) for e in errors) / len(errors) * 100
                med = float(np.median(errors))
                tbs = float(np.mean(tbs_list)) if tbs_list else None
                all_results[ds_name][model][attack] = {
                    'WLA': round(w, 1), 'MedErr': round(med, 1),
                    'TBS': round(tbs, 1) if tbs else None,
                    'N': len(errors), 'tbs_pairs': len(tbs_list)
                }

# Print table
print("=" * 90)
print(f"{'Dataset':<12} {'Model':<18} {'Attack':<14} {'WLA(%)':<10} {'MedErr(km)':<14} {'TBS(km)':<12} {'N':<6} {'TBS_pairs'}")
print("-" * 90)
for ds, ds_data in all_results.items():
    for model, model_data in ds_data.items():
        for attack, m in model_data.items():
            tbs_str = str(m['TBS']) if m['TBS'] is not None else 'N/A'
            print(f"{ds:<12} {model:<18} {attack:<14} {m['WLA']:<10} {m['MedErr']:<14} {tbs_str:<12} {m['N']:<6} {m['tbs_pairs']}")
print("=" * 90)

# Save JSON
with open('paper/table1_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to paper/table1_data.json")
