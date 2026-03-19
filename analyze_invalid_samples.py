import json
import os
import argparse
import collections


def analyze_and_filter(dataset_name, dataset_dir):
    attacks_file = os.path.join(dataset_dir, 'attacks.jsonl')
    if not os.path.exists(attacks_file):
        print(f"  [SKIP] {dataset_name}: {attacks_file} not found")
        return set()
        
    text_counter = collections.Counter()
    
    # 1. Flickr placeholders and generic Web errors
    placeholder_keywords = [
        'no longer available', 'flickr', 'yahoo', 'unavailable', 
        'this photo', 'deleted', 'removed', 'error', 'not found',
        'update your account', '404'
    ]
    
    # 2. Watermarks
    watermark_keywords = [
        '©', 'copyright', 'shutterstock', 'alamy', 'getty', 'images', 
        'istock', 'depositphotos', '123rf', 'dreamstime', 'photocase',
        'photography', 'photo by', 'captured by'
    ]
    
    invalid_base_ids = set()
    total_attacks = 0
    reason_counts = {'placeholder': 0, 'watermark': 0, 'repetition': 0}
    
    with open(attacks_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_attacks += 1
            data = json.loads(line)
            orig_text = str(data.get('original_text', '')).strip().lower()
            
            # Count exact texts to find silent repetitive watermarks (like dates "2008")
            text_counter[orig_text] += 1
            
            is_invalid = False
            base_id = data.get('original_filename', '').split('.')[0]
            
            # Check placeholders
            if any(k in orig_text for k in placeholder_keywords):
                invalid_base_ids.add(base_id)
                reason_counts['placeholder'] += 1
                is_invalid = True
                
            # Check watermarks
            elif any(k in orig_text for k in watermark_keywords):
                invalid_base_ids.add(base_id)
                reason_counts['watermark'] += 1
                is_invalid = True
                
    # Second pass for repetitive texts
    repetitive_texts = set(txt for txt, count in text_counter.items() if count > 5 and txt not in ['stop', 'taxi', 'police', 'bus', 'open', 'p'])
    
    with open(attacks_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            orig_text = str(data.get('original_text', '')).strip().lower()
            base_id = data.get('original_filename', '').split('.')[0]
            
            if base_id not in invalid_base_ids and orig_text in repetitive_texts:
                # E.g. dates like "2009", "2004", camera models
                if orig_text.isdigit() or len(orig_text.split()) >= 2:
                    invalid_base_ids.add(base_id)
                    reason_counts['repetition'] += 1

    print(f"\n[{dataset_name}] Total generated attacks: {total_attacks}")
    print(f"Filtered -> Placeholders: {reason_counts['placeholder']} | Watermarks: {reason_counts['watermark']} | Repetitive Watermarks: {reason_counts['repetition']}")
    print(f"Total Unique Invalid IDs Discarded: {len(invalid_base_ids)}")
    print("Most frequent texts found:")
    for txt, count in text_counter.most_common(10):
        print(f"  [{count}x] {txt}")
        
    return invalid_base_ids


def main():
    parser = argparse.ArgumentParser(description="Analyze and filter invalid samples from attacks.jsonl")
    parser.add_argument("--base-dir", type=str, default="/home/nas/lsr/Data/TRIG-Bench",
                        help="Base directory containing dataset folders (default: /home/nas/lsr/Data/TRIG-Bench)")
    parser.add_argument("--datasets", nargs='+', default=['im2gps3k', 'yfcc4k', 'googlesv'],
                        help="Datasets to analyze (default: im2gps3k yfcc4k googlesv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: invalid_ids.json in script directory)")
    args = parser.parse_args()

    results = {}
    dataset_key_map = {
        'im2gps3k': 'IM2GPS3K',
        'yfcc4k': 'YFCC4K',
        'googlesv': 'GoogleSV',
    }

    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        ds_key = dataset_key_map.get(ds.lower(), ds.upper())
        invalid_ids = analyze_and_filter(ds_key, ds_dir)
        results[ds_key] = list(invalid_ids)

    # Save the output file in the Code directory (current folder)
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invalid_ids.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)

    summary = {k: len(v) for k, v in results.items()}
    print(f"\nSaved invalid IDs to {out_path}: {summary}")


if __name__ == "__main__":
    main()
