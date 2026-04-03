"""
Dataset File Counter — Quick statistics for SIGNPOST-Bench datasets.

Shows file counts for each subdirectory:
  - filtered_images/    (original filtered images)
  - sampled_images/     (sampled street view images)
  - images/Adversarial/ (adversarial attack images)
  - images/Similar/     (similar attack images)
  - images/Random/      (random attack images)
  - images/Blank/       (blank control images)
  - attacks.jsonl       (attack metadata entries)
  - results/            (evaluation result files)

Usage:
    python count_dataset.py                                    # All datasets
    python count_dataset.py --datasets im2gps3k yfcc4k         # Specific datasets
    python count_dataset.py --base-dir /path/to/SIGNPOST-Bench     # Custom path
"""

import os
import json
import argparse


def count_jsonl(filepath):
    """Count lines in a JSONL file."""
    if not os.path.exists(filepath):
        return 0
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
    return count


def count_dir(dirpath, extensions=None):
    """Count files in a directory (non-recursive). Optionally filter by extension."""
    if not os.path.isdir(dirpath):
        return 0
    count = 0
    for fname in os.listdir(dirpath):
        if os.path.isfile(os.path.join(dirpath, fname)):
            if extensions is None or any(fname.lower().endswith(ext) for ext in extensions):
                count += 1
    return count


def get_dir_size_mb(dirpath):
    """Get total size of files in a directory in MB."""
    if not os.path.isdir(dirpath):
        return 0.0
    total = 0
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        if os.path.isfile(fpath):
            total += os.path.getsize(fpath)
    return total / (1024 * 1024)


def count_dataset(dataset_name, dataset_dir):
    """Count all files for a single dataset."""
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    stats = {}
    
    # Original / filtered images
    for subdir in ['filtered_images', 'sampled_images']:
        path = os.path.join(dataset_dir, subdir)
        n = count_dir(path, img_exts)
        size = get_dir_size_mb(path)
        if n > 0:
            stats[subdir] = {'count': n, 'size_mb': size}
    
    # Attack images
    images_dir = os.path.join(dataset_dir, 'images')
    for subdir in ['Adversarial', 'Similar', 'Random', 'Blank']:
        path = os.path.join(images_dir, subdir)
        n = count_dir(path, img_exts)
        size = get_dir_size_mb(path)
        if n > 0:
            stats[f'images/{subdir}'] = {'count': n, 'size_mb': size}
    
    # Metadata files
    attacks_file = os.path.join(dataset_dir, 'attacks.jsonl')
    stats['attacks.jsonl'] = {'count': count_jsonl(attacks_file)}
    
    bench_meta = os.path.join(images_dir, 'benchmark_meta.jsonl')
    if os.path.exists(bench_meta):
        stats['benchmark_meta.jsonl'] = {'count': count_jsonl(bench_meta)}
    
    # Metadata dir
    meta_dir = os.path.join(dataset_dir, 'metadata')
    if os.path.isdir(meta_dir):
        for fname in os.listdir(meta_dir):
            fpath = os.path.join(meta_dir, fname)
            if fname.endswith('.jsonl'):
                stats[f'metadata/{fname}'] = {'count': count_jsonl(fpath)}
            elif fname.endswith('.tsv') or fname.endswith('.csv'):
                # Count lines (minus header)
                with open(fpath, 'r', encoding='utf-8') as f:
                    lines = sum(1 for _ in f) - 1
                stats[f'metadata/{fname}'] = {'count': max(0, lines)}
    
    # Results
    results_dir = os.path.join(dataset_dir, 'results')
    if os.path.isdir(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(results_dir, fname)
                stats[f'results/{fname}'] = {'count': count_jsonl(fpath)}
    
    return stats


def print_dataset_stats(dataset_name, stats):
    """Pretty-print statistics for a dataset."""
    print(f"\n{'='*60}")
    print(f"  📊 {dataset_name}")
    print(f"{'='*60}")
    
    total_images = 0
    total_size = 0.0
    
    # Group by category
    categories = {
        'Original Images': [],
        'Attack Images': [],
        'Metadata': [],
        'Results': [],
    }
    
    for key, val in stats.items():
        count = val.get('count', 0)
        size = val.get('size_mb', 0)
        
        if key in ['filtered_images', 'sampled_images']:
            categories['Original Images'].append((key, count, size))
            total_images += count
            total_size += size
        elif key.startswith('images/'):
            categories['Attack Images'].append((key, count, size))
            total_images += count
            total_size += size
        elif key.startswith('results/'):
            categories['Results'].append((key, count, 0))
        else:
            categories['Metadata'].append((key, count, 0))
    
    for cat_name, items in categories.items():
        if not items:
            continue
        print(f"\n  {cat_name}:")
        for name, count, size in items:
            size_str = f"  ({size:.1f} MB)" if size > 0 else ""
            print(f"    {name:<35} {count:>6} files{size_str}")
    
    print(f"\n  {'─'*40}")
    print(f"  Total images: {total_images:>6}")
    if total_size > 0:
        print(f"  Total size:   {total_size:>6.1f} MB ({total_size/1024:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Count files in SIGNPOST-Bench datasets")
    parser.add_argument("--base-dir", type=str, default="/home/nas/lsr/Data/SIGNPOST-Bench",
                        help="Base directory containing dataset folders")
    parser.add_argument("--datasets", nargs='+', default=None,
                        help="Datasets to count (default: auto-detect all)")
    args = parser.parse_args()
    
    # Auto-detect datasets if not specified
    if args.datasets is None:
        if os.path.isdir(args.base_dir):
            args.datasets = [d for d in sorted(os.listdir(args.base_dir))
                           if os.path.isdir(os.path.join(args.base_dir, d))]
        else:
            print(f"Error: Base directory not found: {args.base_dir}")
            return
    
    if not args.datasets:
        print("No datasets found.")
        return
    
    grand_total_images = 0
    
    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        if not os.path.isdir(ds_dir):
            print(f"\n  [SKIP] {ds}: directory not found ({ds_dir})")
            continue
        
        stats = count_dataset(ds, ds_dir)
        print_dataset_stats(ds.upper(), stats)
        
        # Accumulate grand total
        for key, val in stats.items():
            if key in ['filtered_images', 'sampled_images'] or key.startswith('images/'):
                grand_total_images += val.get('count', 0)
    
    if len(args.datasets) > 1:
        print(f"\n{'='*60}")
        print(f"  🏆 Grand Total: {grand_total_images} images across {len(args.datasets)} datasets")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
