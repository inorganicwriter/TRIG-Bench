import json
import os
import re
import argparse
import collections


def _is_date_string(text: str) -> bool:
    """Check if text is purely a date/timestamp pattern (e.g. '2008', '2019-03-21', '05/12/2008')."""
    text = text.strip()
    if not text:
        return False
    # Pure 4-digit year: "2008", "2019"
    if re.fullmatch(r'(19|20)\d{2}', text):
        return True
    # Date formats: "2019-03-21", "2019/03/21", "03-21-2019", "21.03.2019"
    if re.fullmatch(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}', text):
        return True
    # Date with time: "2019-03-21 14:30:00"
    if re.fullmatch(r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\s+\d{1,2}:\d{2}(:\d{2})?', text):
        return True
    # Month-Year: "Mar 2019", "January 2020"
    if re.fullmatch(r'[a-z]{3,9}\s+(19|20)\d{2}', text):
        return True
    return False


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
        'update your account', '404', 'page not found', 'access denied',
        'image not available', 'content removed', 'violation',
    ]
    
    # 2. Watermarks (including map watermarks and camera brands)
    watermark_keywords = [
        '©', 'copyright', 'shutterstock', 'alamy', 'getty', 'images', 
        'istock', 'depositphotos', '123rf', 'dreamstime', 'photocase',
        'photography', 'photo by', 'captured by',
        # Map watermarks
        'google', 'mapbox', 'openstreetmap', 'esri', 'here.com',
        'tomtom', 'baidu', '百度', '高德', '腾讯地图', 'amap',
        'imagery', 'map data', '© openstreetmap',
        # Camera / device metadata text
        'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
        'samsung', 'apple', 'iphone', 'huawei', 'xiaomi', 'oppo',
        # Car brand logos (common in street view panoramas)
        '汽车', '轿车', '客车', 'motor', 'automobile',
    ]
    
    # 3. Date/timestamp patterns (photo capture dates burned into image)
    date_keywords = [
        'date:', 'taken on', 'captured on', 'shot on',
    ]
    
    # 4. Image failure / placeholder messages
    failure_keywords = [
        'this image', 'no image', 'broken image', 'loading',
        'thumbnail', 'preview', 'placeholder', 'default',
        'coming soon', 'under construction',
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
            
            # Check date/timestamp patterns (e.g. "2008-05-12", "2019/03/21")
            elif any(k in orig_text for k in date_keywords):
                invalid_base_ids.add(base_id)
                reason_counts.setdefault('date_stamp', 0)
                reason_counts['date_stamp'] += 1
                is_invalid = True
            elif _is_date_string(orig_text):
                invalid_base_ids.add(base_id)
                reason_counts.setdefault('date_stamp', 0)
                reason_counts['date_stamp'] += 1
                is_invalid = True
            
            # Check image failure messages
            elif any(k in orig_text for k in failure_keywords):
                invalid_base_ids.add(base_id)
                reason_counts.setdefault('failure_msg', 0)
                reason_counts['failure_msg'] += 1
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


def _extract_base_id_from_filename(fname):
    """
    Extract the base_id from a synthesized image filename.
    
    Naming convention from main_benchmark.py:
      {base_name}_{attack_type}_{safe_text}.png   (attack images)
      {base_name}_Blank.png                        (blank images)
    
    base_name = original_filename without extension (e.g. "12345" from "12345.jpg")
    
    We need to match base_name which may itself contain underscores,
    so we use the attacks.jsonl to build a lookup of known base_ids.
    """
    # Remove extension
    name_no_ext = os.path.splitext(fname)[0]
    return name_no_ext


def delete_invalid_files(dataset_dir, invalid_ids, dry_run=False):
    """
    Delete ALL files related to invalid sample IDs.
    
    Deletes:
    - Synthesized attack images in images/{Adversarial,Similar,Random,Blank}/
    - Filtered original images in filtered_images/
    - Sampled original images in sampled_images/
    - Evaluation result entries in results/*.jsonl
    - Benchmark metadata entries in images/benchmark_meta.jsonl
    - Removes entries from attacks.jsonl (rewrites the file)
    
    Returns: dict with deletion counts
    """
    counts = {
        'attack_images_deleted': 0,
        'original_images_deleted': 0,
        'attacks_removed': 0,
        'results_removed': 0,
        'meta_removed': 0,
    }
    
    if not invalid_ids:
        return counts
    
    # Build a set of known base_ids from attacks.jsonl for accurate filename matching
    attacks_file = os.path.join(dataset_dir, 'attacks.jsonl')
    # Map: base_id -> list of expected output filenames
    expected_filenames = {}  # base_id -> set of filenames (without dir)
    if os.path.exists(attacks_file):
        with open(attacks_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                original_filename = data.get('original_filename', '')
                base_name = os.path.splitext(original_filename)[0]
                if base_name not in expected_filenames:
                    expected_filenames[base_name] = set()
                
                # Blank image
                expected_filenames[base_name].add(f"{base_name}_Blank.png")
                
                # Attack images
                attack_dict = data.get('attacks', {})
                for attack_type, attack_text in attack_dict.items():
                    safe_text = "".join([c for c in attack_text if c.isalnum() or c in (' ', '_', '-')]).strip()
                    save_name = f"{base_name}_{attack_type}_{safe_text}.png"
                    if len(save_name.encode('utf-8')) > 255:
                        max_text_len = 255 - len(f"{base_name}_{attack_type}_.png".encode('utf-8'))
                        safe_text = safe_text[:max(10, max_text_len)]
                        save_name = f"{base_name}_{attack_type}_{safe_text}.png"
                    expected_filenames[base_name].add(save_name)
    
    # Collect all filenames that belong to invalid IDs
    invalid_filenames = set()
    for base_id in invalid_ids:
        if base_id in expected_filenames:
            invalid_filenames.update(expected_filenames[base_id])
    
    # Build prefix set for fast O(1) fallback matching
    invalid_prefixes = set(base_id + '_' for base_id in invalid_ids)
    
    # 1. Delete ALL synthesized attack images whose original base_id is invalid
    #    Logic: if the original image is invalid, ALL generated images from it are invalid
    images_dir = os.path.join(dataset_dir, 'images')
    for subdir in ['Adversarial', 'Similar', 'Random', 'Blank']:
        subdir_path = os.path.join(images_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        file_list = os.listdir(subdir_path)
        subdir_deleted = 0
        print(f"  Scanning {subdir}/ ({len(file_list)} files)...")
        for fname in file_list:
            # Check if this file belongs to any invalid base_id
            # Method: try all possible prefixes split by '_'
            should_delete = fname in invalid_filenames
            if not should_delete:
                parts = fname.split('_')
                prefix = ''
                for i, part in enumerate(parts[:-1]):
                    prefix = prefix + part + '_' if prefix else part + '_'
                    if prefix in invalid_prefixes:
                        should_delete = True
                        break
            
            if should_delete:
                fpath = os.path.join(subdir_path, fname)
                if not dry_run:
                    os.remove(fpath)
                subdir_deleted += 1
        
        counts['attack_images_deleted'] += subdir_deleted
        action = "Would delete" if dry_run else "Deleted"
        print(f"    {action}: {subdir_deleted}/{len(file_list)} files")
    
    # 2. Delete original images that are invalid OR have no corresponding attack
    #    (filtered_images with no entry in attacks.jsonl are useless for evaluation)
    
    # Build set of base_ids that have valid attacks (not in invalid_ids)
    valid_attack_ids = set()
    if os.path.exists(attacks_file):
        with open(attacks_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                base_id = data.get('original_filename', '').split('.')[0]
                if base_id not in invalid_ids:
                    valid_attack_ids.add(base_id)
    
    for img_subdir in ['filtered_images', 'sampled_images']:
        img_dir = os.path.join(dataset_dir, img_subdir)
        if not os.path.isdir(img_dir):
            continue
        file_list = os.listdir(img_dir)
        subdir_deleted = 0
        print(f"  Scanning {img_subdir}/ ({len(file_list)} files)...")
        for fname in file_list:
            file_base_id = os.path.splitext(fname)[0]
            # Delete if: explicitly invalid OR no corresponding attack entry
            should_delete = (file_base_id in invalid_ids) or (file_base_id not in valid_attack_ids)
            if should_delete:
                fpath = os.path.join(img_dir, fname)
                if not dry_run:
                    os.remove(fpath)
                subdir_deleted += 1
        
        counts['original_images_deleted'] += subdir_deleted
        action = "Would delete" if dry_run else "Deleted"
        print(f"    {action}: {subdir_deleted}/{len(file_list)} files (keeping {len(file_list) - subdir_deleted})")
    
    # 3. Rewrite attacks.jsonl without invalid entries
    if os.path.exists(attacks_file):
        valid_lines = []
        removed = 0
        with open(attacks_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                base_id = data.get('original_filename', '').split('.')[0]
                if base_id in invalid_ids:
                    removed += 1
                else:
                    valid_lines.append(line)
        
        counts['attacks_removed'] = removed
        
        if not dry_run and removed > 0:
            backup_path = attacks_file + '.bak'
            if not os.path.exists(backup_path):
                os.rename(attacks_file, backup_path)
            else:
                os.remove(attacks_file)
            with open(attacks_file, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            print(f"  Rewrote {attacks_file}: {removed} entries removed, {len(valid_lines)} kept")
            print(f"  Backup: {backup_path}")
        elif dry_run and removed > 0:
            print(f"  [DRY-RUN] Would remove {removed} entries from {attacks_file}")
    
    # 4. Rewrite benchmark_meta.jsonl without invalid entries
    bench_meta = os.path.join(images_dir, 'benchmark_meta.jsonl')
    if os.path.exists(bench_meta):
        valid_lines = []
        removed = 0
        with open(bench_meta, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                orig_src = data.get('original_source', '')
                base_id = os.path.splitext(orig_src)[0]
                if base_id in invalid_ids:
                    removed += 1
                else:
                    valid_lines.append(line)
        
        counts['meta_removed'] = removed
        
        if not dry_run and removed > 0:
            backup_path = bench_meta + '.bak'
            if not os.path.exists(backup_path):
                os.rename(bench_meta, backup_path)
            else:
                os.remove(bench_meta)
            with open(bench_meta, 'w', encoding='utf-8') as f:
                f.writelines(valid_lines)
            print(f"  Rewrote benchmark_meta.jsonl: {removed} entries removed")
        elif dry_run and removed > 0:
            print(f"  [DRY-RUN] Would remove {removed} entries from benchmark_meta.jsonl")
    
    # 5. Clean evaluation results
    results_dir = os.path.join(dataset_dir, 'results')
    if os.path.isdir(results_dir):
        for result_file in os.listdir(results_dir):
            if not result_file.endswith('.jsonl'):
                continue
            result_path = os.path.join(results_dir, result_file)
            valid_lines = []
            removed = 0
            with open(result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Result files may use 'filename', 'image_id', or 'photo_id'
                        file_ref = data.get('filename', data.get('image_id', data.get('photo_id', '')))
                        base_id = os.path.splitext(file_ref)[0].split('_')[0] if file_ref else ''
                        if base_id in invalid_ids:
                            removed += 1
                        else:
                            valid_lines.append(line)
                    except json.JSONDecodeError:
                        valid_lines.append(line)
            
            counts['results_removed'] += removed
            
            if not dry_run and removed > 0:
                with open(result_path, 'w', encoding='utf-8') as f:
                    f.writelines(valid_lines)
                print(f"  Cleaned {result_file}: {removed} entries removed")
            elif dry_run and removed > 0:
                print(f"  [DRY-RUN] Would remove {removed} entries from {result_file}")
    
    return counts


def main():
    parser = argparse.ArgumentParser(description="Analyze and filter invalid samples from attacks.jsonl")
    parser.add_argument("--base-dir", type=str, default="/home/nas/lsr/Data/SIGNPOST-Bench",
                        help="Base directory containing dataset folders (default: /home/nas/lsr/Data/SIGNPOST-Bench)")
    parser.add_argument("--datasets", nargs='+', default=['im2gps3k', 'yfcc4k', 'googlesv'],
                        help="Datasets to analyze (default: im2gps3k yfcc4k googlesv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: invalid_ids.json in script directory)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete invalid sample files (images + attacks.jsonl entries)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting (use with --delete)")
    args = parser.parse_args()

    results = {}
    dataset_key_map = {
        'im2gps3k': 'IM2GPS3K',
        'yfcc4k': 'YFCC4K',
        'googlesv': 'GoogleSV',
        'baidusv': 'BaiduSV',
    }

    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        ds_key = dataset_key_map.get(ds.lower(), ds.upper())
        invalid_ids = analyze_and_filter(ds_key, ds_dir)
        results[ds_key] = list(invalid_ids)
        
        # Delete invalid files if requested
        if args.delete and invalid_ids:
            print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Deleting invalid samples for {ds_key}...")
            counts = delete_invalid_files(ds_dir, invalid_ids, dry_run=args.dry_run)
            print(f"  Attack images deleted: {counts['attack_images_deleted']}")
            print(f"  Original images deleted: {counts['original_images_deleted']}")
            print(f"  Attack entries removed: {counts['attacks_removed']}")
            print(f"  Benchmark meta removed: {counts['meta_removed']}")
            print(f"  Result entries removed: {counts['results_removed']}")

    # Save the output file in the Code directory (current folder)
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'invalid_ids.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)

    summary = {k: len(v) for k, v in results.items()}
    print(f"\nSaved invalid IDs to {out_path}: {summary}")
    
    if not args.delete:
        print(f"\n💡 To actually delete these files, re-run with --delete:")
        print(f"   python analyze_invalid_samples.py --datasets {' '.join(args.datasets)} --delete --dry-run  # preview")
        print(f"   python analyze_invalid_samples.py --datasets {' '.join(args.datasets)} --delete            # execute")


if __name__ == "__main__":
    main()
