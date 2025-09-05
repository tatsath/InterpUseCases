#!/bin/bash

# Working Delphi Auto-Interp Script with OpenRouter
# ================================================
# - Based on the working run_minimal_delphi.sh
# - Explainer: GPT-3.5-turbo via OpenRouter
# - Dataset: wikitext
# - Scoring: detection (F1)
# - Output: feature_number, label, f1_score

echo "🚀 Running Working Delphi Auto-Interp with OpenRouter"
echo "====================================================="

# Check for OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY not set!"
    echo "Please set your OpenRouter API key:"
    echo "export OPENROUTER_API_KEY='your_api_key_here'"
    exit 1
fi

echo "✅ OpenRouter API key found"

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
RUN_NAME="working_delphi_openrouter"

# Run Delphi with working parameters
cd /home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/delphi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m delphi \
  "$MODEL" \
  "$SAE_PATH" \
  --n_tokens 10000 \
  --max_latents 5 \
  --hookpoints layers.16 \
  --scorers detection \
  --explainer_model "openai/gpt-3.5-turbo" \
  --explainer_provider "openrouter" \
  --explainer_model_max_len 512 \
  --dataset_repo wikitext \
  --dataset_name wikitext-103-raw-v1 \
  --dataset_split "train[:1%]" \
  --filter_bos \
  --name "$RUN_NAME"

# Extract and format results
echo ""
echo "📊 Results Summary:"
echo "==================="
echo "Feature | Label | F1_Score"
echo "--------|-------|---------"

# Parse results from the run directory
RUN_DIR="runs/$RUN_NAME"
if [ -d "$RUN_DIR" ]; then
    # Look for explanation files
    for file in "$RUN_DIR"/explanations/*.json; do
        if [ -f "$file" ]; then
            # Extract feature number from filename
            feature_num=$(basename "$file" | grep -o '[0-9]\+')
            
            # Extract label and score from JSON
            label=$(python3 -c "
import json
try:
    with open('$file', 'r') as f:
        data = json.load(f)
        print(data.get('label', 'N/A'))
except:
    print('N/A')
")
            
            # Look for corresponding score file
            score_file="$RUN_DIR/scores/detection/$(basename "$file" .json).txt"
            f1_score="N/A"
            if [ -f "$score_file" ]; then
                f1_score=$(python3 -c "
import json
try:
    with open('$score_file', 'r') as f:
        data = json.load(f)
        print(f\"{data.get('f1_score', 0.0):.3f}\")
except:
    print('N/A')
")
            fi
            
            printf "%-7s | %-20s | %s\n" "$feature_num" "$label" "$f1_score"
        fi
    done
else
    echo "❌ Run directory not found: $RUN_DIR"
fi

echo ""
echo "✅ Analysis complete!"
echo "📁 Full results in: $RUN_DIR"
