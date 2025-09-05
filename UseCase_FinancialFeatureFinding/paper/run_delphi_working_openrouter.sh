#!/bin/bash

# Working Delphi Auto-Interp Script with OpenRouter and FAISS
# ===========================================================
# - Based on the working run_minimal_delphi.sh
# - Explainer: GPT-3.5-turbo via OpenRouter
# - Dataset: wikitext
# - Scoring: detection (F1)
# - Non-activating examples: 50 examples using FAISS similarity search
# - Output: feature_number, label, f1_score

echo "🚀 Running Working Delphi Auto-Interp with OpenRouter and FAISS"
echo "==============================================================="

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

# Generate versioned run name
BASE_NAME="working_delphi_openrouter"
VERSION=1
RUN_NAME="${BASE_NAME}_v${VERSION}"

# Find the next available version number
while [ -d "results/${BASE_NAME}_v${VERSION}" ] || [ -f "results/${BASE_NAME}_v${VERSION}_summary.csv" ]; do
    VERSION=$((VERSION + 1))
    RUN_NAME="${BASE_NAME}_v${VERSION}"
done

echo "📝 Using run name: $RUN_NAME"

# Run Delphi with working parameters
cd /home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/delphi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m delphi \
  "$MODEL" \
  "$SAE_PATH" \
  --n_tokens 100000 \
  --max_latents 5 \
  --hookpoints layers.16 \
  --scorers detection \
  --explainer_model "openai/gpt-3.5-turbo" \
  --explainer_provider "openrouter" \
  --explainer_model_max_len 512 \
  --num_gpus 4 \
  --num_examples_per_scorer_prompt 1 \
  --n_non_activating 50 \
  --non_activating_source "FAISS" \
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
RUN_DIR="results/$RUN_NAME"

if [ -d "$RUN_DIR" ]; then
    echo "📊 Results found, generating summary..."
    
    # Call CSV generation script with error handling
    if /home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/paper/generate_csv_results.sh "$RUN_NAME" "$RUN_DIR"; then
        echo "✅ CSV generation completed successfully!"
        CSV_FILE="results/${RUN_NAME}_summary.csv"
        echo "📄 CSV summary saved to: $CSV_FILE"
    else
        echo "❌ CSV generation failed, but analysis results are available in: $RUN_DIR"
        echo "📁 You can manually check the results in the explanations and scores directories"
    fi
else
    echo "❌ Run directory not found: $RUN_DIR"
    echo "📁 Check if the analysis completed successfully"
fi

echo ""
echo "✅ Analysis complete!"
echo "📁 Full results in: $RUN_DIR"
