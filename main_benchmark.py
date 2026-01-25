import os
import argparse
import random
from PIL import Image
import torch
from benchmark_engine.relevance_scorer import SemanticRelevanceEngine
from benchmark_engine.text_injector import VisualAttacker
from benchmark_engine.distractor_pool import CANDIDATE_CITIES

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Generator (Quantitative)")
    parser.add_argument("--clean-dir", type=str, required=True, help="Directory containing clean images (Step 1 output)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save final benchmark images")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Initialize Engines
    print("Initializing Engines...")
    try:
        scorer = SemanticRelevanceEngine(model_name=args.clip_model)
        attacker = VisualAttacker() # Will auto-find font or use default
    except Exception as e:
        print(f"Failed to initialize engines: {e}")
        return

    # 2. Load Images
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = [f for f in os.listdir(args.clean_dir) if f.lower().endswith(valid_exts)]
    print(f"Found {len(image_files)} clean images.")

    # 3. Process Loop
    processed_count = 0
    for i, filename in enumerate(image_files):
        if args.limit > 0 and processed_count >= args.limit:
            break
            
        print(f"\n[{i+1}/{len(image_files)}] Processing {filename}...")
        image_path = os.path.join(args.clean_dir, filename)
        
        try:
            original_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        # A. Relevance Scoring
        print(f"  > Calculating relevance against {len(CANDIDATE_CITIES)} candidates...")
        # To speed up, we might randomly sample candidates if list is huge, but 50-100 is fine.
        categorized = scorer.classify_distractors(original_img, CANDIDATE_CITIES)
        
        # B. Selection Strategy: Orthogonal Matrix
        # Semantic Dimension: Hard (Parallel), Mid, Easy (Orthogonal)
        selected_texts = [] # list of (level, text, score)
        
        if categorized['hard']:
            selected_texts.append(('hard', categorized['hard'][0][0], categorized['hard'][0][1]))
        if categorized['mid']:
            selected_texts.append(('mid', categorized['mid'][0][0], categorized['mid'][0][1]))
        if categorized['easy']:
            selected_texts.append(('easy', categorized['easy'][0][0], categorized['easy'][0][1]))

        # C. Generation Loop (Cartesian Product)
        # We try to apply BOTH "Object" and "Random" strategies to ALL text levels if possible.
        base_name = os.path.splitext(filename)[0]
        
        for sem_level, text, score in selected_texts:
            # Strategies to try: 
            # 1. Random/Center (Background/Fixed) -> Corresponds to Level 2/3 Physical
            # 2. Object (Moving Object) -> Corresponds to Level 1 Physical
            
            strategies = ["random", "object"]
            
            for strategy in strategies:
                # Need to handle Object detection failure inside VisualAttacker?
                # VisualAttacker handles fallback to random, but we should label it correctly.
                # However, for the benchmark dataset, we want distinct files.
                
                target_cls = ['car', 'bus', 'truck', 'person'] if strategy == "object" else None
                
                # We interpret "random" here as general background/fixed placement
                
                attack_img = attacker.inject_text(
                    original_img, 
                    text, 
                    position=strategy, 
                    target_class=target_cls
                )
                
                # Save Image
                save_name = f"{base_name}_{sem_level}_{strategy}_{text}.jpg"
                save_path = os.path.join(args.output_dir, save_name)
                attack_img.save(save_path, quality=95)
                print(f"      -> Saved: {save_name} (Sem: {sem_level}, Phy: {strategy})")

                # Save Metadata
                meta_entry = {
                    "filename": save_name,
                    "original_source": filename,
                    "injected_text": text,
                    "semantic_difficulty": sem_level, # hard/mid/easy
                    "relevance_score": float(score),
                    "physical_level": "Level 1" if strategy == "object" else "Level 2/3",
                    "injection_strategy": strategy
                }
                with open(os.path.join(args.output_dir, "benchmark_meta.jsonl"), "a", encoding="utf-8") as f:
                    import json
                    f.write(json.dumps(meta_entry) + "\n")

        processed_count += 1

    print(f"\nBenchmark Generation Complete. Metadata saved to {os.path.join(args.output_dir, 'benchmark_meta.jsonl')}")

if __name__ == "__main__":
    main()
