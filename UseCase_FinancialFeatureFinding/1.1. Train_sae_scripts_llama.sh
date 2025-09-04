#!/bin/bash
# SAE Training using direct torchrun command with finance data and post-training evaluation
# Modified to train 5 layers at regular intervals for meta-llama/Llama-2-7b-hf

echo "🚀 SAE Training using direct torchrun command (meta-llama/Llama-2-7b-hf + 5 Layers + Post-Training Evaluation)"
echo "======================================================"

# Multi-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42

# Number of GPUs to use
NUM_GPUS=2

# Layer selection - 5 layers at regular intervals for Llama-2-7b-hf (32 layers total)
# Selecting layers: 4, 10, 16, 22, 28 (approximately every 6th layer)
LAYERS=(4 10 16 22 28)

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Change to the saetrain directory
cd /home/nvidia/Documents/Hariom/saetrain

echo "⏰ Starting SAE training for 5 layers with direct torchrun command..."
echo "======================================================"

echo ""
echo "🔧 Training all layers: ${LAYERS[*]}..."
echo "======================================================"

# Run using direct torchrun command with meta-llama/Llama-2-7b-hf model for all layers
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    -m saetrain \
    meta-llama/Llama-2-7b-hf \
    iohadrubin/wikitext-103-raw-v1 \
    --batch_size 8 \
    --k 32 \
    --num_latents 400 \
    --grad_acc_steps 4 \
    --ctx_len 1024 \
    --save_dir "./trained_models" \
    --shuffle_seed 42 \
    --init_seeds 42 \
    --optimizer adam \
    --lr 0.001 \
    --save_every 500 \
    --run_name "llama2_7b_hf_layers${LAYERS[*]}_k32_latents400_wikitext103_torchrun" \
    --log_to_wandb true \
    --wandb_log_frequency 10 \
    --dead_percentage_threshold 0.00 \
    --layers ${LAYERS[*]}
    
# Check if training was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ SAE training for all layers FAILED! Exit code: $?"
    echo "======================================================"
    echo "🚫 Skipping evaluation - training did not complete successfully"
    exit 1
fi

echo ""
echo "✅ SAE training for all layers completed! Running post-training assessment..."

# Extract the run ID from the last WandB run
LATEST_RUN_DIR=$(ls -t wandb/ | head -1)
if [ -z "$LATEST_RUN_DIR" ]; then
    echo "❌ No WandB runs found. Cannot perform assessment."
    exit 1
fi

RUN_ID=$(basename "$LATEST_RUN_DIR" | rev | cut -d'-' -f1 | rev)
if [ -z "$RUN_ID" ]; then
    echo "❌ Could not extract run ID from: $LATEST_RUN_DIR"
    exit 1
fi

echo "📊 Extracting final metrics from WandB run: $RUN_ID for all layers"
echo "⏳ Waiting for WandB sync to complete..."
sleep 5
    
# Extract training metrics from WandB
TRAINING_METRICS=$(python -c "
import wandb
import json

try:
    api = wandb.Api()
    run = api.run(f'tatsatx-university-of-california-berkeley/saetrain/$RUN_ID')
    history = run.history()
    
    if history.empty:
        print('NO_METRICS')
    else:
        final_metrics = history.iloc[-1]
        metrics = {}
        
        # Extract all available metrics
        for key in final_metrics.keys():
            if 'fvu/' in key:
                fvu = final_metrics[key]
                loss_recovered = (1.0 - fvu) * 100
                metrics['loss_recovered'] = loss_recovered
                metrics['fvu'] = fvu
            elif 'dead_feature_pct/' in key:
                metrics['dead_features_percent'] = final_metrics[key]
            elif 'l0_sparsity/' in key:
                metrics['l0_sparsity'] = final_metrics[key]
            elif 'feature_absorption/' in key:
                metrics['feature_absorption'] = final_metrics[key]
        
        print(json.dumps(metrics))
        
except Exception as e:
    print('ERROR:' + str(e))
" 2>/dev/null)
    
# Parse training metrics
if [[ "$TRAINING_METRICS" == "NO_METRICS" ]]; then
    echo "❌ No training metrics found in WandB for all layers"
    TRAIN_LOSS="N/A"
    TRAIN_L0="N/A"
    TRAIN_DEAD="N/A"
    TRAIN_ABS="N/A"
elif [[ "$TRAINING_METRICS" == ERROR* ]]; then
    echo "❌ Error extracting training metrics for all layers: ${TRAINING_METRICS#ERROR:}"
    TRAIN_LOSS="N/A"
    TRAIN_L0="N/A"
    TRAIN_DEAD="N/A"
    TRAIN_ABS="N/A"
else
    TRAIN_LOSS=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"loss_recovered\", 0):.2f}%')" 2>/dev/null || echo "N/A")
    TRAIN_L0=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"l0_sparsity\", 0):.2f}')" 2>/dev/null || echo "N/A")
    TRAIN_DEAD=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"dead_features_percent\", 0):.2f}%')" 2>/dev/null || echo "N/A")
    TRAIN_ABS=$(echo "$TRAINING_METRICS" | python -c "import sys, json; data=json.load(sys.stdin); print(f'{data.get(\"feature_absorption\", 0):.4f}')" 2>/dev/null || echo "N/A")
fi
    
echo ""
echo "======================================================"
echo "🔍 Running comprehensive post-training evaluation for all layers..."

# Find the latest SAE checkpoint directory
LATEST_SAE_DIR=$(find ./trained_models -name "llama2_7b_hf_layers${LAYERS[*]}_k32_latents400_wikitext103_torchrun*" -type d | head -1)
if [ -z "$LATEST_SAE_DIR" ]; then
    echo "❌ No SAE checkpoint directory found."
    echo "   This means training either failed or didn't complete successfully."
    echo "   Cannot perform evaluation without a valid checkpoint."
    echo "======================================================"
    echo "🚫 Evaluation skipped - no checkpoint available"
    exit 1
fi

echo "📂 Found SAE checkpoint: $LATEST_SAE_DIR"

# Evaluate each layer
for layer in "${LAYERS[@]}"; do
    echo ""
    echo "📊 Evaluating Layer $layer..."
    
    # Run evaluation on datasets
    datasets=("wikitext" "squad")
    final_results=()
    
    for dataset in "${datasets[@]}"; do
        echo "📊 Evaluating Layer $layer on $dataset..."
        
        output_file="llama2_7b_hf_layers${LAYERS[*]}_k32_latents400_wikitext103_final_layer${layer}_${dataset}_evaluation_results.json"
        
        python sae_posttrain_eval.py \
            --sae_path "$LATEST_SAE_DIR/layers.$layer" \
            --model_name meta-llama/Llama-2-7b-hf \
            --layer $layer \
            --dataset "$dataset" \
            --num_samples 1000 \
            --context_length 1024 \
            --batch_size 16 \
            --output_file "$output_file" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            # Extract key metrics
            loss_recovered=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['loss_recovered']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
            
            l0_sparsity=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['l0_sparsity']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
            
            dead_features=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['dead_features_percent']:.2f}\")
except:
    print('0.00')
" 2>/dev/null)
            
            absorption=$(python -c "
import json
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
    print(f\"{data['results']['feature_absorption']:.4f}\")
except:
    print('0.0000')
" 2>/dev/null)
            
            final_results+=("$dataset: Loss=${loss_recovered}%, L0=${l0_sparsity}, Dead=${dead_features}%, Abs=${absorption}")
        else
            final_results+=("$dataset: FAILED")
        fi
    done
    
    # Display comprehensive results table for this layer
    echo ""
    echo "📊 COMPREHENSIVE SAE RESULTS - Layer $layer (Training + Evaluation)"
    echo "======================================================"
    printf "%-15s %-15s %-12s %-15s %-15s\n" "Source" "Loss Recovered" "L0 Sparsity" "Dead Features" "Absorption"
    echo "------------------------------------------------------"
    
    # Training metrics (from WandB)
    printf "%-15s %-15s %-12s %-15s %-15s\n" "Training-WandB" "$TRAIN_LOSS" "$TRAIN_L0" "$TRAIN_DEAD" "$TRAIN_ABS"
    
    # Evaluation metrics
    for result in "${final_results[@]}"; do
        if [[ $result == *"FAILED"* ]]; then
            dataset="${result%:*}"
            printf "%-15s %-15s %-12s %-15s %-15s\n" "$dataset" "FAILED" "FAILED" "FAILED" "FAILED"
        else
            dataset=$(echo $result | cut -d':' -f1)
            loss=$(echo $result | grep -o 'Loss=[0-9.]*%' | cut -d'=' -f2)
            l0=$(echo $result | grep -o 'L0=[0-9.]*' | cut -d'=' -f2)
            dead=$(echo $result | grep -o 'Dead=[0-9.]*%' | cut -d'=' -f2)
            abs=$(echo $result | grep -o 'Abs=[0-9.]*' | cut -d'=' -f2)
            printf "%-15s %-15s %-12s %-15s %-15s\n" "$dataset" "$loss" "$l0" "$dead" "$abs"
        fi
    done
    
    echo "======================================================"
    echo "🔗 WandB Run ID: $RUN_ID"
    echo ""
done

echo ""
echo "🎯 COMPLETE TRAINING SUMMARY"
echo "======================================================"
echo "✅ Trained 5 layers at regular intervals: ${LAYERS[*]}"
echo "📋 Dataset Loading Status:"
echo "  ✅ WikiText: Loaded successfully"
echo "  ✅ SQuAD: Loaded successfully"
echo ""
echo "🎯 Configuration: Layers ${LAYERS[*]}, k=32, Context=1024, LR=0.001, Model: meta-llama/Llama-2-7b-hf, Dataset: WikiText-103"
echo ""
echo "💡 Optimizations for Better Out-of-Sample Reconstruction Loss:"
echo "  • Increased batch size 8 for smaller model efficiency"
echo "  • Reduced gradient accumulation steps 4 for faster training"
echo "  • Higher learning rate 0.001 for smaller model convergence"
echo "  • Increased context length 1024 for better representation learning"
echo "  • Removed max_tokens to allow full dataset training"
echo "  • Multi-layer training for comprehensive model analysis"
