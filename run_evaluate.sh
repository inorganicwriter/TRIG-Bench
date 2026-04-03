#!/bin/bash
# ==============================================
# Multi-Dataset + Multi-Model Evaluation Script
# ==============================================
#
# Usage (in tmux, conda activate trig):
#
#   # Local model (no API key needed):
#   bash run_evaluate.sh <model> [dataset]
#
#   # Cloud API model (API key via argument or env var):
#   bash run_evaluate.sh <model> [dataset] [api_key]
#
#   # Using environment variables (recommended for security):
#   export OPENROUTER_API_KEY=sk-or-xxx
#   export SILICONFLOW_API_KEY=sk-sf-xxx
#   bash run_evaluate.sh gpt-4o im2gps3k
#
# Examples:
#   bash run_evaluate.sh qwen3-30b                                    # Local vLLM, all datasets
#   bash run_evaluate.sh qwen3-8b yfcc4k                              # Local vLLM, single dataset
#   bash run_evaluate.sh gpt-4o im2gps3k sk-or-xxx                    # OpenRouter: GPT-4o
#   bash run_evaluate.sh qwen2.5-vl-72b-sf googlesv sk-sf-xxx         # SiliconFlow: Qwen2.5-VL-72B
#   bash run_evaluate.sh claude-sonnet-4.6 all sk-or-xxx              # OpenRouter: Claude Sonnet 4.6
#
# Available models (short names from api_client.py MODEL_REGISTRY):
#
#   Local vLLM:   qwen3-30b, qwen3-8b
#
#   SiliconFlow:  qwen3-vl-235b-thinking-sf, qwen3-vl-235b-sf,
#                 qwen3-vl-32b-thinking-sf, qwen3-vl-32b-sf,
#                 qwen3-vl-30b-thinking-sf, qwen3-vl-30b-sf,
#                 qwen3-vl-8b-thinking-sf, qwen3-vl-8b-sf,
#                 qwen2.5-vl-72b-sf, qwen2.5-vl-32b-sf,
#                 glm-4.6v-sf, glm-4.5v-sf
#
#   OpenRouter:   gpt-5, gpt-5.4, gpt-5.4-mini, gpt-4.1, gpt-4.1-mini, gpt-4o,
#                 o3, o4-mini,
#                 claude-opus-4.6, claude-sonnet-4.6, claude-haiku-4.5,
#                 claude-3.7-sonnet, claude-3.7-sonnet-thinking,
#                 gemini-3.1-pro, gemini-3.1-flash, gemini-3-pro,
#                 gemini-2.5-pro, gemini-2.5-flash,
#                 grok-4, grok-4.20,
#                 qwen3-vl-235b-or, qwen3-vl-235b-thinking-or,
#                 qwen3-vl-32b-or, qwen3-vl-30b-or, qwen3-vl-30b-thinking-or,
#                 qwen3-vl-8b-or, qwen3-vl-8b-thinking-or,
#                 qwen3.5-397b-or, qwen3.5-122b-or, qwen3.5-27b-or,
#                 llama-4-maverick, llama-4-scout,
#                 mistral-medium-3.1, glm-4.6v, glm-4.5v,
#                 nova-pro, nova-lite
#
#   Free (OpenRouter, $0, vision-capable, for testing):
#                 gemma3-27b-free, gemma3-12b-free, gemma3-4b-free,
#                 mistral-small-3.1-free, nemotron-nano-12b-vl-free
#
# Datasets: yfcc4k, im2gps3k, googlesv, baidusv
#   (default: all four when no dataset is specified, or use "all")
#
# API Key Resolution Order:
#   1. Command-line argument (3rd positional arg)
#   2. Environment variables:
#      - OPENROUTER_API_KEY  (for openrouter models)
#      - SILICONFLOW_API_KEY (for siliconflow models)
#      - RELAY_API_KEY       (for relay/LCPU models)
#      - OPENAI_API_KEY      (for openai models)
#   3. Auto-detected (local vLLM uses default "local-key", no key needed)
# ==============================================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash run_evaluate.sh <model> [dataset|all] [api_key]"
    echo ""
    echo "Local models:       qwen3-30b, qwen3-8b"
    echo "SiliconFlow models: qwen3-vl-235b-sf, qwen3-vl-32b-sf, qwen2.5-vl-72b-sf,"
    echo "                    glm-4.6v-sf, glm-4.5v-sf  (suffix: -sf)"
    echo "OpenRouter models:  gpt-5, gpt-4o, o3, o4-mini,"
    echo "                    claude-opus-4.6, claude-sonnet-4.6,"
    echo "                    gemini-2.5-pro, gemini-2.5-flash,"
    echo "                    grok-4, llama-4-maverick  (and more)"
    echo ""
    echo "Datasets: yfcc4k, im2gps3k, googlesv (default: all three)"
    echo ""
    echo "API Key (optional, can also use env vars):"
    echo "  export OPENROUTER_API_KEY=sk-or-xxx"
    echo "  export SILICONFLOW_API_KEY=sk-sf-xxx"
    echo ""
    echo "Full model list: python -m evaluation.api_client"
    exit 1
fi

MODEL=$1

# Dataset selection: "all" or empty = all three datasets
if [ $# -ge 2 ] && [ "$2" != "all" ]; then
    DATASETS=("$2")
else
    DATASETS=("yfcc4k" "im2gps3k" "googlesv" "baidusv")
fi

# API key: from argument or environment variable
API_KEY_ARG=""
if [ $# -ge 3 ]; then
    API_KEY_ARG="--api-key $3"
fi

cd /home/nas/lsr/Code/TRIG-Bench

echo "=========================================="
echo "  Evaluation: ${DATASETS[*]} x ${MODEL}"
echo "  Time: $(date)"
echo "=========================================="

for DS in "${DATASETS[@]}"; do
    echo ""
    echo ">>> Dataset: ${DS} | Model: ${MODEL}"
    echo "    $(date)"
    python run_pipeline.py --dataset "$DS" --stage evaluate --model "$MODEL" $API_KEY_ARG
    echo "    ${DS} done. $(date)"
done

echo ""
echo "=========================================="
echo "  All evaluations complete!"
echo "  Time: $(date)"
echo "=========================================="
