"""
Cleanup script: Delete orphaned images from the incorrect 50-char truncation run.

These files were generated with safe_text[:50] truncation, which differs from
the correct behavior (only truncate when filename > 255 bytes).

This script scans the attacks.jsonl to find entries where the attack text is
longer than 50 chars, then checks if a truncated (50-char) version of the
file exists on disk. If found, it deletes the orphan.

Usage:
    python cleanup_truncated.py --attacks-file /path/to/attacks.jsonl --images-dir /path/to/images
    python cleanup_truncated.py --attacks-file /path/to/attacks.jsonl --images-dir /path/to/images --dry-run
"""

import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned truncated-filename images")
    parser.add_argument("--attacks-file", type=str, required=True, help="Path to attacks.jsonl")
    parser.add_argument("--images-dir", type=str, required=True, help="Path to images output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted, don't actually delete")
    args = parser.parse_args()

    attacks = []
    with open(args.attacks_file, 'r', encoding='utf-8') as f:
        for line in f:
            attacks.append(json.loads(line))

    deleted = 0
    checked = 0

    for entry in attacks:
        original_filename = entry.get('original_filename', '')
        base_name = os.path.splitext(original_filename)[0]
        attack_dict = entry.get('attacks', {})

        for attack_type, attack_text in attack_dict.items():
            safe_text_full = "".join([c for c in attack_text if c.isalnum() or c in (' ', '_', '-')]).strip()
            
            # Only check entries where truncation at 50 chars would differ
            if len(safe_text_full) <= 50:
                continue
            
            checked += 1
            
            # The old (wrong) truncated filename
            safe_text_50 = safe_text_full[:50]
            subdir = attack_type.capitalize()
            old_filename = f"{base_name}_{attack_type}_{safe_text_50}.png"
            old_path = os.path.join(args.images_dir, subdir, old_filename)
            
            if os.path.exists(old_path):
                if args.dry_run:
                    print(f"[DRY-RUN] Would delete: {old_path}")
                else:
                    os.remove(old_path)
                    print(f"Deleted: {old_path}")
                deleted += 1

    print(f"\n--- Summary ---")
    print(f"Entries with text > 50 chars: {checked}")
    print(f"{'Would delete' if args.dry_run else 'Deleted'}: {deleted}")

if __name__ == "__main__":
    main()
