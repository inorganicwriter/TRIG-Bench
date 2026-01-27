import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize TRIG-Bench Multi-Model Benchmark")
    # Allow multiple inputs in format: ModelName=Path
    parser.add_argument("--results", nargs='+', required=True, 
                        help="List of results files in format 'ModelName=path/to/results.jsonl' (e.g. Qwen=res_qwen.jsonl GPT4=res_gpt4.jsonl)")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="Directory to save plots")
    return parser.parse_args()

def load_data(results_args):
    all_data = []
    
    for item in results_args:
        if '=' not in item:
            print(f"Error: Format should be ModelName=FilePath, got '{item}'")
            continue
            
        model_name, file_path = item.split('=', 1)
        print(f"Loading {model_name} from {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"  -> File not found: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    filename = entry['filename']
                    
                    # Parse Attack Type
                    attack_type = "unknown"
                    if "_similar_" in filename: attack_type = "similar"
                    elif "_random_" in filename: attack_type = "random"
                    elif "_adversarial_" in filename: attack_type = "adversarial"
                    elif "_clean" in filename: attack_type = "clean"
                    
                    entry['model'] = model_name
                    entry['attack_type'] = attack_type
                    all_data.append(entry)
                except:
                    pass
                    
    return pd.DataFrame(all_data)

def plot_leaderboard(df, output_dir):
    """Overall Leaderboard (MGD - Mean Geodesic Distance)"""
    plt.figure(figsize=(10, 6))
    
    # Calculate global mean error per model
    leaderboard = df.groupby("model")["error_km"].mean().sort_values().reset_index()
    
    sns.barplot(data=leaderboard, x="model", y="error_km", palette="viridis")
    plt.title("Leaderboard: Mean Geodesic Distance (Lower is Better)")
    plt.ylabel("Avg Error (km)")
    
    save_path = os.path.join(output_dir, "benchmark_leaderboard.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Leaderboard to {save_path}")
    plt.close()

def plot_robustness_drop(df, output_dir):
    """Comparison of Clean vs Adversarial Error per Model"""
    plt.figure(figsize=(12, 6))
    
    # Filter for aggregated Adversarial vs Clean
    # We treat Similar/Random/Adversarial all as "Attacked" for high level summary, or split them?
    # Let's show specific attack types
    
    avg_errors = df.groupby(["model", "attack_type"])["error_km"].mean().reset_index()
    
    # Order: Clean first, then others
    order = ["clean", "similar", "random", "adversarial"]
    
    sns.barplot(data=avg_errors, x="model", y="error_km", hue="attack_type", 
                hue_order=order, palette="rocket_r")
    
    plt.title("Model Robustness Under Different Attacks")
    plt.ylabel("Avg Error (km)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    save_path = os.path.join(output_dir, "benchmark_robustness_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved Robustness Comparison to {save_path}")
    plt.close()

def plot_cdf_comparison(df, output_dir):
    """CDF comparison for Adversarial samples only (Models vs Models)"""
    plt.figure(figsize=(10, 6))
    
    # Only look at Adversarial for this plot to see who defends best
    adv_df = df[df['attack_type'] == 'adversarial'].dropna(subset=['error_km'])
    
    if not adv_df.empty:
        sns.ecdfplot(data=adv_df, x="error_km", hue="model", linewidth=2)
        plt.title("CDF: Performance on Adversarial Samples")
        plt.xscale('log')
        plt.xlabel("Error (km) - Log Scale")
        plt.ylabel("Cumulative Probability")
        
        save_path = os.path.join(output_dir, "benchmark_cdf_adversarial.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved Adversarial CDF to {save_path}")
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = load_data(args.results)
    if df.empty:
        print("No data loaded.")
        return

    print(f"Loaded {len(df)} records from {df['model'].nunique()} models.")

    # 1. Overall Leaderboard
    plot_leaderboard(df, args.output_dir)
    
    # 2. Detailed Attack Comparison
    plot_robustness_drop(df, args.output_dir)
    
    # 3. CDF Comparison
    plot_cdf_comparison(df, args.output_dir)

if __name__ == "__main__":
    main()
