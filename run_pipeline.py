import os
import sys
import subprocess
import argparse
from pathlib import Path

# ================= Configuration =================
# Fixed Base Paths (Hardcoded as requested)
RAW_DATA_ROOT = Path("/home/nas/lsr/Data/dataset")
GOOGLESV_ROOT = Path("/home/nas/lsr/Data/GoogleSV")
BAIDUSV_ROOT = Path("/home/nas/lsr/BaiduSvs_history/output")
TRIG_BENCH_ROOT = Path("/home/nas/lsr/Data/SIGNPOST-Bench")
CODE_DIR = Path("/home/nas/lsr/Code/SIGNPOST-Bench")

# Services
LOCAL_API_BASE = "http://0.0.0.0:8001/v1"
DEFAULT_MODEL = "qwen3-30b"
COMFY_SERVER = "127.0.0.1:8188"

# Import unified model registry from api_client
# This avoids duplicating model definitions across files
try:
    sys.path.insert(0, str(CODE_DIR))
    from evaluation.api_client import MODEL_REGISTRY, PROVIDER_CONFIGS
    _registry_loaded = True
except ImportError:
    _registry_loaded = False
    MODEL_REGISTRY = {}
    PROVIDER_CONFIGS = {}


def resolve_model(model_arg):
    """
    Resolve model short name to (model_path, short_name, api_base).
    Uses the unified MODEL_REGISTRY from evaluation/api_client.py.
    Falls back to local vLLM for unknown names.
    """
    if model_arg in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_arg]
        provider = entry.get("provider", "local")
        provider_cfg = PROVIDER_CONFIGS.get(provider, {})
        api_base = provider_cfg.get("api_base", LOCAL_API_BASE)
        return entry["model"], model_arg, api_base
    # If full path given, assume local vLLM
    base = model_arg.rstrip('/').split('/')[-1].lower()
    return model_arg, base, LOCAL_API_BASE

def parse_args():
    parser = argparse.ArgumentParser(description="Run SIGNPOST-Bench Pipeline for a specific dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., yfcc4k, im2gps3k)")
    parser.add_argument("--stage", type=str, choices=['all', 'attack_gen', 'synthesize', 'evaluate'], default='all', 
                        help="Run a specific stage. 'attack_gen' includes filtering, metadata, and LLM generation. Default: all")
    parser.add_argument("--api-key", type=str, default=None, help="API Key (default: None, uses env vars for cloud providers)")
    parser.add_argument("--api-base", type=str, default=None, help="Override API base URL")
    parser.add_argument("--raw-img-dir", type=str, default=None, 
                        help="Override raw image directory (default: auto-derived from dataset name)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model for evaluation (short name or full path). Short names: " + 
                             ", ".join(MODEL_REGISTRY.keys()))
    return parser.parse_args()

# Dataset-specific image directory names
# When the image subdirectory doesn't match the dataset name
IMAGE_DIR_OVERRIDES = {
    "im2gps3k": "im2gps3ktest",
    # Add more here as needed, e.g.:
    # "streetview": "images",
}

def get_paths(dataset_name, raw_img_dir_override=None):
    # Derived Paths based on dataset name
    # Supports override for datasets with non-standard directory structures
    
    work_dir = TRIG_BENCH_ROOT / dataset_name
    
    if raw_img_dir_override:
        raw_img_dir = Path(raw_img_dir_override)
    elif dataset_name == "googlesv":
        # GoogleSV uses sampled images in the SIGNPOST-Bench work directory
        raw_img_dir = work_dir / "sampled_images"
    elif dataset_name == "baidusv":
        # BaiduSV uses sampled & cropped images in the SIGNPOST-Bench work directory
        raw_img_dir = work_dir / "sampled_images"
    else:
        img_subdir = IMAGE_DIR_OVERRIDES.get(dataset_name, dataset_name)
        raw_img_dir = RAW_DATA_ROOT / dataset_name / img_subdir
    
    # Metadata CSV location
    if dataset_name == "googlesv":
        raw_meta_csv = work_dir / "googlesv_metadata_address.csv"
    elif dataset_name == "baidusv":
        raw_meta_csv = work_dir / "baidusv_metadata.csv"
    else:
        raw_meta_csv = RAW_DATA_ROOT / dataset_name / f"{dataset_name}_metadata_address.csv"
    
    return {
        "raw_img_dir": raw_img_dir,
        "raw_meta_csv": raw_meta_csv,
        "work_dir": work_dir,
        "metadata_dir": work_dir / "metadata",
        "images_dir": work_dir / "images",
        "results_dir": work_dir / "results",
        "attacks_file": work_dir / "attacks.jsonl",
        "dataset_name": dataset_name
    }


def run_step(step_name, command, cwd=CODE_DIR):
    print(f"\n{'='*10} Step: {step_name} {'='*10}")
    print(f"Running: {' '.join(str(x) for x in command)}")
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {step_name}: {e}")
        sys.exit(1)

def ensure_dirs(paths):
    print(f"Creating directories in {paths['work_dir']}...")
    os.makedirs(paths['metadata_dir'], exist_ok=True)
    os.makedirs(paths['images_dir'], exist_ok=True)
    os.makedirs(paths['results_dir'], exist_ok=True)

def main():
    args = parse_args()
    paths = get_paths(args.dataset, raw_img_dir_override=args.raw_img_dir)
    
    # Verify input exists (Only strictly enforced for attack_gen stage, others rely on previous outputs)
    if args.stage in ['all', 'attack_gen'] and not paths['raw_img_dir'].exists():
        print(f"Error: Raw image directory not found: {paths['raw_img_dir']}")
        sys.exit(1)
        
    ensure_dirs(paths)

    # Resolve model
    if args.model:
        model_path, model_short, model_api_base = resolve_model(args.model)
    else:
        model_path, model_short, model_api_base = DEFAULT_MODEL, "qwen3-30b", LOCAL_API_BASE
    # Allow CLI override of api-base
    api_base = args.api_base if args.api_base else model_api_base
    
    # Derived file paths helper
    filtered_img_dir = paths['work_dir'] / "filtered_images"
    clean_meta = paths['metadata_dir'] / f"{paths['dataset_name']}_clean_meta.jsonl"
    gt_tsv = paths['metadata_dir'] / f"{paths['dataset_name']}_gt.tsv"
    bench_meta = paths['images_dir'] / "benchmark_meta.jsonl"

    # ================= Stage 1: Attack Generation (Filter + Metadata + LLM) =================
    if args.stage in ['all', 'attack_gen']:
        print(">>> Stage: Attack Generation (Filter -> Metadata -> LLM)")
        
        # Step 0: Filter Images
        run_step("Filter Images (OCR)", [
            sys.executable, "data_collector/filter_images.py",
            "--input-dir", str(paths['raw_img_dir']),
            "--output-dir", str(filtered_img_dir),
            "--gpu"
        ])

        # Step 1: Prepare Metadata
        run_step("Prepare Metadata", [
            sys.executable, "convert_metadata.py",
            "--csv", str(paths['raw_meta_csv']),
            "--out-dir", str(paths['metadata_dir']),
            "--dataset-name", paths['dataset_name']
        ])
        
        # Step 2: Generate Attacks
        print(">>> Generating Attacks with LLM...")
        if not filtered_img_dir.exists():
             print(f"Warning: Filtered images dir {filtered_img_dir} not found.")
        
        attack_cmd = [
            sys.executable, "data_collector/generate_attacks.py",
            "--clean-meta", str(clean_meta),
            "--original-dir", str(filtered_img_dir),
            "--output", str(paths['attacks_file']),
            "--api-base", api_base,
            "--model", model_path,
        ]
        if args.api_key:  # FIX: only pass --api-key when explicitly provided
            attack_cmd.extend(["--api-key", args.api_key])
        run_step("Generate Attacks", attack_cmd)

    # ================= Stage 2: Synthesis =================
    if args.stage in ['all', 'synthesize']:
        print(">>> Stage: Image Synthesis (ComfyUI)")

        if not paths['attacks_file'].exists():
            print(f"Error: Attack file {paths['attacks_file']} not found. Run 'attack' stage first.")
            sys.exit(1)

        run_step("Synthesize Images", [
            sys.executable, "main_benchmark.py",
            "--attack-file", str(paths['attacks_file']),
            "--output-dir", str(paths['images_dir']),
            "--comfy-server", COMFY_SERVER
        ])

    # ================= Stage 4: Evaluation =================
    if args.stage in ['all', 'evaluate']:
        print(f">>> Stage: Evaluation (Model: {model_short})")
        
        # Step 1: Evaluate Original Images (baseline for TBS)
        original_result_file = paths['results_dir'] / f"results_Original_{model_short}.jsonl"
        
        # Build common eval args (api-key only appended when not None)
        def _build_eval_cmd(img_dir, output_file, bench_meta_path=None, baseline_path=None):
            cmd = [
                sys.executable, "evaluate.py",
                "--img-dir", str(img_dir),
                "--metadata-file", str(gt_tsv),
                "--output", str(output_file),
                "--model", model_short,   # FIX: pass short name, not model_path
                "--api-base", api_base,
            ]
            if args.api_key:             # FIX: only pass --api-key when explicitly provided
                cmd.extend(["--api-key", args.api_key])
            if bench_meta_path:
                cmd.extend(["--bench-meta", str(bench_meta_path)])
            if baseline_path and Path(baseline_path).exists():
                cmd.extend(["--baseline", str(baseline_path)])
            return cmd

        if filtered_img_dir.exists() and filtered_img_dir.is_dir():
            print(f"\n--- Evaluating Original Images ({model_short}) ---")
            run_step(f"Evaluate Original ({model_short})",
                     _build_eval_cmd(filtered_img_dir, original_result_file))
        else:
            print(f"Warning: Filtered images dir {filtered_img_dir} not found, TBS will be unavailable.")

        # Step 2: Evaluate Attack Images (with baseline for TBS)
        subdirs = ["Adversarial", "Similar", "Random"]

        for subdir in subdirs:
            target_dir = paths['images_dir'] / subdir
            if target_dir.exists() and target_dir.is_dir():
                print(f"\n--- Evaluating Subdirectory: {subdir} ({model_short}) ---")
                result_file = paths['results_dir'] / f"results_{subdir}_{model_short}.jsonl"
                run_step(f"Evaluate {subdir} ({model_short})",
                         _build_eval_cmd(target_dir, result_file,
                                         bench_meta_path=bench_meta,
                                         baseline_path=original_result_file))
            else:
                print(f"Skipping evaluation for {subdir} (Directory not found)")

    print(f"\nPipeline execution for stage '{args.stage}' completed!")

    print(f"\nPipeline completed successfully! Check results in {paths['results_dir']}")

if __name__ == "__main__":
    main()
