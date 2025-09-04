#!/bin/bash

echo "🚀 Running Delphi Auto-Interp on Llama-2-7B SAE Models"
echo "========================================================"

# Check if Delphi is installed
if ! python -c "import delphi" 2>/dev/null; then
    echo "📦 Installing Delphi dependencies..."
    
    # Install required packages
    pip install "eai-sparsify>=1.1.3" datasets faiss-cpu sentence-transformers vllm orjson
    
    # Clone and install Delphi
    if [ ! -d "delphi" ]; then
        echo "📥 Cloning Delphi repository..."
        git clone https://github.com/EleutherAI/sae-auto-interp delphi
    fi
    
    echo "🔧 Installing Delphi..."
    cd delphi && pip install -e . && cd ..
fi

echo "✅ Delphi is ready!"

# Define paths
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"

# Check if SAE exists
if [ ! -d "$SAE_PATH" ]; then
    echo "❌ SAE not found at: $SAE_PATH"
    echo "Please check the path and ensure the SAE model is available."
    exit 1
fi

echo "🎯 Base Model: $BASE_MODEL"
echo "🎯 SAE Path: $SAE_PATH"
echo "🎯 Layer: 16"

echo ""
echo "🚀 Running Delphi Auto-Interp..."
echo "Command: python -m delphi $BASE_MODEL $SAE_PATH --n_tokens 50000 --max_latents 20 --hookpoints layers.16 --scorers detection --filter_bos --example_ctx_len 16 --min_examples 20 --n_non_activating 5 --dataset_repo iohadrubin/wikitext-103-raw-v1 --dataset_split train --num_gpus 4 --name llama2_7b_layer16_sae_autointerp_4gpu_csv"

# Run Delphi with WikiText-103 dataset using 4 GPUs for ultra-fast testing (FAISS disabled, very short labels)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python -m delphi \
    "$BASE_MODEL" \
    "$SAE_PATH" \
    --n_tokens 50000 \
    --max_latents 20 \
    --hookpoints layers.16 \
    --scorers detection \
    --filter_bos \
    --example_ctx_len 16 \
    --min_examples 20 \
    --n_non_activating 5 \
    --dataset_repo "iohadrubin/wikitext-103-raw-v1" \
    --dataset_split "train" \
    --num_gpus 4 \
    --name "llama2_7b_layer16_sae_autointerp_4gpu_csv"

echo ""
echo "🎉 Delphi Auto-Interp completed!"
echo "📁 Check the results in: llama2_7b_layer16_sae_autointerp_4gpu_csv/"
echo ""
echo "💡 Note: FAISS disabled due to compatibility issues"
echo "  - Using random non-activating examples instead"
echo "  - This should avoid the 'NonActivatingExample' error"
echo "  - Short labels enabled with: --example_ctx_len 16, --min_examples 20"
echo "  - Using WikiText-103 dataset (same as SAE training)"
echo "  - Using 4 GPUs for faster processing"
echo "  - Using default explainer model (optimized for Delphi)"
echo "  - Results will be saved in the specified output directory"

# Post-process results to create CSV file
echo ""
echo "🔄 Creating CSV output file..."
OUTPUT_DIR="llama2_7b_layer16_sae_autointerp_4gpu_csv"
CSV_FILE="results/$OUTPUT_DIR/llama2_7b_layer16_sae_autointerp_results.csv"

if [ -d "results/$OUTPUT_DIR/explanations" ]; then
    echo "📊 Creating CSV with columns: layer, feature_id, label"
    
    # Create CSV header
    echo "layer,feature_id,label" > "$CSV_FILE"
    
    # Process each explanation file
    for file in results/$OUTPUT_DIR/explanations/*.txt; do
        if [ -f "$file" ]; then
            # Extract layer and latent info from filename (e.g., layers.16_latent0.txt)
            filename=$(basename "$file" .txt)
            layer=$(echo "$filename" | cut -d'_' -f1)
            feature_id=$(echo "$filename" | cut -d'_' -f2)
            
            # Read the label content and clean it up
            label=$(cat "$file" | tr -d '\n' | sed 's/^"//;s/"$//' | sed 's/^ma"//')
            
            # Add to CSV
            echo "$layer,$feature_id,$label" >> "$CSV_FILE"
            echo "✅ Processed: $filename"
        fi
    done
    
    echo "📁 CSV file created: $CSV_FILE"
    echo "📊 CSV contents:"
    cat "$CSV_FILE"
else
    echo "❌ Explanations directory not found. CSV creation skipped."
fi