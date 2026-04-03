"""
Stratified Sampling Script for Google Street View Dataset.

Samples images proportionally by country from the panoids.csv metadata,
then copies/symlinks sampled images to a flat output directory for pipeline processing.

Usage:
    python sample_googlesv.py \
        --metadata /home/nas/lsr/Data/GoogleSV/metadata/panoids.csv \
        --images-root /home/nas/lsr/Data/GoogleSV/images \
        --output-dir /home/nas/lsr/Data/SIGNPOST-Bench/googlesv/sampled_images \
        --sample-rate 0.01 \
        --output-csv /home/nas/lsr/Data/SIGNPOST-Bench/googlesv/googlesv_metadata_address.csv
"""

import os
import csv
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Stratified Sampling for Google Street View")
    parser.add_argument("--metadata", type=str, required=True, 
                        help="Path to panoids.csv")
    parser.add_argument("--images-root", type=str, required=True, 
                        help="Root directory of images (e.g. /home/nas/lsr/Data/GoogleSV/images)")
    parser.add_argument("--output-dir", type=str, required=True, 
                        help="Directory to copy/symlink sampled images (flat structure)")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Output CSV with sampled metadata (pipeline-compatible)")
    parser.add_argument("--sample-rate", type=float, default=0.01,
                        help="Sampling rate per country (default: 0.01 = 1%%)")
    parser.add_argument("--min-per-country", type=int, default=10,
                        help="Minimum samples per country (default: 10)")
    parser.add_argument("--max-total", type=int, default=0,
                        help="Maximum total samples (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying (saves disk space)")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Read Metadata and Group by Country
    print(f"Reading metadata from {args.metadata}...")
    by_country = defaultdict(list)
    total_rows = 0
    
    with open(args.metadata, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            country = row.get('country', 'Unknown')
            by_country[country].append(row)
    
    print(f"Total records: {total_rows}")
    print(f"Countries: {len(by_country)}")
    
    # 2. Stratified Sampling
    sampled = []
    for country, rows in sorted(by_country.items()):
        n = max(args.min_per_country, int(len(rows) * args.sample_rate))
        n = min(n, len(rows))  # Can't sample more than available
        selected = random.sample(rows, n)
        sampled.extend(selected)
        print(f"  {country}: {len(rows)} -> {n} sampled")
    
    # Cap total if specified
    if args.max_total > 0 and len(sampled) > args.max_total:
        random.shuffle(sampled)
        sampled = sampled[:args.max_total]
        print(f"\nCapped total to {args.max_total}")
    
    print(f"\nTotal sampled: {len(sampled)}")
    
    # 3. Copy/Symlink Images and Build Output CSV
    print(f"\n{'Symlinking' if args.symlink else 'Copying'} images to {output_dir}...")
    
    success = 0
    missing = 0
    output_rows = []
    
    for row in tqdm(sampled, desc="Processing"):
        panoid = row.get('panoid', '')
        angle = row.get('angle', '')
        country = row.get('country', '')
        city = row.get('city', '')
        lat = row.get('lat', '')
        lon = row.get('lon', '')
        
        # Construct source path: images/{country}/{city}/{panoid}_{angle}.jpg
        src_filename = f"{panoid}_{angle}.jpg"
        src_path = images_root / country / city / src_filename
        
        if not src_path.exists():
            missing += 1
            continue
        
        # Destination: flat directory with unique name
        # Use panoid_angle as unique identifier
        dst_path = output_dir / src_filename
        
        try:
            if args.symlink:
                if dst_path.exists() or dst_path.is_symlink():
                    dst_path.unlink()
                dst_path.symlink_to(src_path)
            else:
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
            
            success += 1
            output_rows.append({
                'photo_id': f"{panoid}_{angle}",
                'panoid': panoid,
                'latitude': lat,
                'longitude': lon,
                'country': country,
                'city': city,
                'angle': angle,
                'source_path': str(src_path)
            })
        except Exception as e:
            print(f"Error: {src_path} -> {e}")
            missing += 1
    
    # 4. Save Output CSV
    print(f"\nSaving metadata to {args.output_csv}...")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['photo_id', 'panoid', 'latitude', 'longitude', 
                                                'country', 'city', 'angle', 'source_path'])
        writer.writeheader()
        writer.writerows(output_rows)
    
    # Summary
    print(f"\n{'='*40}")
    print(f"Sampling Complete!")
    print(f"  Total metadata records: {total_rows}")
    print(f"  Sampled: {len(sampled)}")
    print(f"  Successfully processed: {success}")
    print(f"  Missing files: {missing}")
    print(f"  Images saved to: {output_dir}")
    print(f"  Metadata saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
