"""
classify_taxonomy.py — Auto-classify original scene text into T1/T2/T3 tiers.

Tier definitions:
  T1 (Portable):    Global brands, product names, watermarks, camera text, dates, URLs.
                    Could appear on any continent. No geographic signal.
  T2 (Cultural):    Non-Latin scripts, generic local business names, language-specific text.
                    Narrows to a cultural/linguistic region but not a specific city.
  T3 (Geo-Specific): Street names, city names, postal codes, named landmarks, addresses.
                    Directly identifies a geographic entity.

Usage:
  python classify_taxonomy.py --datasets im2gps3k yfcc4k googlesv
"""

import json
import os
import re
import argparse
import unicodedata
from collections import Counter

# ==================== Classification Rules ====================

# T1: Portable / Global brands / Watermarks / Camera metadata
T1_EXACT = {
    'stop', 'exit', 'open', 'closed', 'push', 'pull', 'enter', 'no entry',
    'danger', 'warning', 'caution', 'slow', 'yield', 'one way',
    'men', 'women', 'restroom', 'toilet', 'wc',
    'wifi', 'atm', 'parking', 'p', 'info', 'taxi',
}

T1_KEYWORDS = [
    # Global brands
    'coca-cola', 'pepsi', 'mcdonald', 'starbucks', 'subway', 'kfc',
    'nike', 'adidas', 'samsung', 'apple', 'sony', 'canon', 'nikon',
    'fedex', 'dhl', 'ups', 'amazon', 'google', 'microsoft',
    'toyota', 'honda', 'bmw', 'mercedes', 'ford', 'volkswagen',
    'shell', 'bp', 'total', 'esso', 'mobil',
    'visa', 'mastercard', 'paypal',
    'hilton', 'marriott', 'hyatt',
    'reconyx',  # Camera brand (common in YFCC)
    # Watermarks / metadata
    '©', 'copyright', 'all rights reserved', 'photo by', 'captured by',
    'shutterstock', 'getty', 'alamy', 'istock', 'dreamstime',
    'www.', 'http', '.com', '.net', '.org',
    # Generic signs
    'speed limit', 'no parking', 'keep out', 'do not',
    'for sale', 'for rent', 'for lease',
]

# T3: Geo-specific patterns
T3_PATTERNS = [
    # Street/road names
    r'\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|lane|ln|drive|dr|way|place|pl|court|ct)\b',
    r'\b(?:strasse|straße|str\.|gasse|weg|platz|allee)\b',  # German
    r'\b(?:rue|avenue|boulevard|place|chemin)\b',  # French
    r'\b(?:calle|avenida|paseo|plaza|camino)\b',  # Spanish
    r'\b(?:via|viale|piazza|corso|largo)\b',  # Italian
    # Postal codes
    r'\b\d{5}(?:-\d{4})?\b',  # US ZIP
    r'\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b',  # UK postcode
    r'\b\d{3}-\d{4}\b',  # Japan postal
    # City/country names (common)
    r'\b(?:paris|london|tokyo|berlin|rome|madrid|moscow|beijing|shanghai|new york|los angeles|chicago|sydney|melbourne|toronto|vancouver|dubai|singapore|hong kong|bangkok|seoul|mumbai|delhi|cairo|istanbul|athens|vienna|prague|budapest|warsaw|amsterdam|brussels|zurich|lisbon|stockholm|oslo|helsinki|copenhagen|dublin|edinburgh)\b',
    # Explicit address patterns
    r'\b\d+\s+\w+\s+(?:st|street|rd|road|ave|avenue)\b',
    # Named landmarks
    r'\b(?:station|gare|bahnhof|estación|stazione)\b',
    r'\b(?:airport|aeroporto|flughafen|aéroport|aeropuerto)\b',
    r'\b(?:university|universidad|université|universität|大学)\b',
    r'\b(?:museum|musée|museo|国立|市立)\b',
    r'\b(?:church|cathedral|mosque|temple|shrine|basilica|寺|神社)\b',
]

# Non-Latin script detection for T2
def has_non_latin_script(text):
    """Check if text contains significant non-Latin characters (CJK, Arabic, Cyrillic, etc.)."""
    non_latin_count = 0
    total_alpha = 0
    for ch in text:
        if ch.isalpha():
            total_alpha += 1
            cat = unicodedata.category(ch)
            name = unicodedata.name(ch, '')
            if any(s in name for s in ['CJK', 'ARABIC', 'CYRILLIC', 'HANGUL', 'THAI',
                                       'DEVANAGARI', 'BENGALI', 'TAMIL', 'TELUGU',
                                       'KATAKANA', 'HIRAGANA', 'HEBREW', 'GEORGIAN']):
                non_latin_count += 1
    if total_alpha == 0:
        return False
    return non_latin_count / total_alpha > 0.3


def classify_text(original_text, text_location=""):
    """Classify a single text entry into T1, T2, or T3."""
    text_lower = original_text.strip().lower()
    
    # Empty or very short
    if len(text_lower) < 2:
        return 'T1', 'too_short'
    
    # Pure numbers (dates, phone numbers, etc.) -> T1
    if re.match(r'^[\d\s\-\.\/:]+$', text_lower):
        return 'T1', 'numeric'
    
    # Exact match T1
    if text_lower in T1_EXACT:
        return 'T1', 'generic_sign'
    
    # T1 keyword match
    for kw in T1_KEYWORDS:
        if kw in text_lower:
            return 'T1', f'brand_or_watermark ({kw})'
    
    # T3: Check geo-specific patterns
    for pattern in T3_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return 'T3', f'geo_pattern'
    
    # T2: Non-Latin script
    if has_non_latin_script(original_text):
        return 'T2', 'non_latin_script'
    
    # T2: Location description heuristics from text_location field
    loc_lower = text_location.lower() if text_location else ""
    if any(kw in loc_lower for kw in ['storefront', 'shop', 'store', 'restaurant', 'hotel',
                                       'cafe', 'bar', 'market', 'pharmacy']):
        # Local business name -> T2
        return 'T2', 'local_business'
    
    # Default: if it's a short word on a sign, likely T2 (culturally ambiguous)
    if len(text_lower.split()) <= 2:
        return 'T2', 'short_ambiguous'
    
    # Longer text that didn't match T1 or T3 -> T2
    return 'T2', 'default_cultural'


def process_dataset(dataset_name, dataset_dir):
    """Process a single dataset's attacks.jsonl and classify all entries."""
    attacks_file = os.path.join(dataset_dir, 'attacks.jsonl')
    if not os.path.exists(attacks_file):
        print(f"  Skipping {dataset_name}: {attacks_file} not found")
        return []
    
    results = []
    tier_counter = Counter()
    
    with open(attacks_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                orig_text = data.get('original_text', '')
                text_loc = data.get('text_location', '')
                base_id = data.get('original_filename', '').split('.')[0]
                
                tier, reason = classify_text(orig_text, text_loc)
                tier_counter[tier] += 1
                
                results.append({
                    'base_id': base_id,
                    'original_text': orig_text,
                    'text_location': text_loc,
                    'tier': tier,
                    'reason': reason,
                    'adversarial_text': data.get('attacks', {}).get('adversarial', ''),
                    'dataset': dataset_name,
                })
            except Exception as e:
                pass
    
    # Save per-dataset taxonomy
    output_file = os.path.join(dataset_dir, 'taxonomy_labels.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n  [{dataset_name}] Total: {sum(tier_counter.values())}")
    print(f"    T1 (Portable):     {tier_counter.get('T1', 0)}")
    print(f"    T2 (Cultural):     {tier_counter.get('T2', 0)}")
    print(f"    T3 (Geo-Specific): {tier_counter.get('T3', 0)}")
    print(f"    Saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify scene text into T1/T2/T3 taxonomy")
    parser.add_argument('--datasets', nargs='+', default=['im2gps3k', 'yfcc4k', 'googlesv'],
                        help='Dataset names to process')
    
    # Linux Server Data Directory
    SERVER_DATA_DIR = "/home/nas/lsr/Data/TRIG-Bench"
    parser.add_argument('--base-dir', type=str, default=SERVER_DATA_DIR,
                        help='Base directory containing dataset folders')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  TRIG-Bench Scene-Text Taxonomy Classifier")
    print("=" * 50)
    
    all_results = []
    for ds in args.datasets:
        ds_dir = os.path.join(args.base_dir, ds)
        results = process_dataset(ds, ds_dir)
        all_results.extend(results)
    
    # Global summary
    global_counter = Counter(r['tier'] for r in all_results)
    print(f"\n{'=' * 50}")
    print(f"  GLOBAL SUMMARY ({len(all_results)} total entries)")
    print(f"{'=' * 50}")
    print(f"  T1 (Portable):     {global_counter.get('T1', 0)} ({100*global_counter.get('T1',0)/max(len(all_results),1):.1f}%)")
    print(f"  T2 (Cultural):     {global_counter.get('T2', 0)} ({100*global_counter.get('T2',0)/max(len(all_results),1):.1f}%)")
    print(f"  T3 (Geo-Specific): {global_counter.get('T3', 0)} ({100*global_counter.get('T3',0)/max(len(all_results),1):.1f}%)")
    
    # Print examples for each tier
    for tier in ['T1', 'T2', 'T3']:
        examples = [r for r in all_results if r['tier'] == tier][:5]
        print(f"\n  {tier} Examples:")
        for ex in examples:
            print(f"    \"{ex['original_text'][:40]}\" ({ex['reason']})")


if __name__ == '__main__':
    main()
