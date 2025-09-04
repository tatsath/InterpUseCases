import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
import json
import os
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from collections import defaultdict
import seaborn as sns

def load_sae_model(sae_path, layer_name):
    """Load SAE model from path"""
    sae_file = os.path.join(sae_path, layer_name, "sae.safetensors")
    cfg_file = os.path.join(sae_path, layer_name, "cfg.json")
    
    # Load config
    with open(cfg_file, 'r') as f:
        sae_config = json.load(f)
    
    # Load weights
    sae_weights = {}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with safe_open(sae_file, framework="pt", device=device) as f:
        for key in f.keys():
            sae_weights[key] = f.get_tensor(key)
    
    encoder_weight = sae_weights.get('encoder.weight', None)
    decoder_weight = sae_weights.get('W_dec', None)
    encoder_bias = sae_weights.get('encoder.bias', None)
    decoder_bias = sae_weights.get('b_dec', None)
    
    return {
        'encoder_weight': encoder_weight,
        'decoder_weight': decoder_weight,
        'encoder_bias': encoder_bias,
        'decoder_bias': decoder_bias,
        'config': sae_config
    }

def get_layer_activations(text, model, tokenizer, target_layer, max_length=1024):
    """Get activations from target layer for given text"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Get activations from target layer (0-indexed)
        layer_activations = hidden_states[target_layer]
        
    return layer_activations

def encode_activations(activations, encoder_weight, encoder_bias=None):
    """Encode activations using the trained SAE"""
    # Ensure all tensors are on the same device
    device = encoder_weight.device
    activations = activations.to(device).to(encoder_weight.dtype)
    
    if encoder_bias is not None:
        encoder_bias = encoder_bias.to(device)
        encoded = torch.nn.functional.linear(activations, encoder_weight, encoder_bias)
    else:
        encoded = torch.nn.functional.linear(activations, encoder_weight, encoder_weight)
        
    encoded = torch.nn.functional.relu(encoded)
    return encoded

def create_financial_test_sentences():
    """Create comprehensive financial test sentences"""
    financial_sentences = {
        'earnings_reports': [
            "The company reported quarterly earnings of $2.5 billion, exceeding analyst expectations.",
            "Revenue growth accelerated to 25% year-over-year in the latest quarter.",
            "Profit margins expanded due to cost-cutting measures and pricing power.",
            "The company raised its full-year guidance based on strong demand trends.",
            "Earnings per share increased 30% compared to the same period last year."
        ],
        'stock_market': [
            "The stock market rallied today with the S&P 500 gaining 2.5% on strong earnings reports.",
            "Tech stocks led the market higher as investors bet on AI and cloud computing growth.",
            "Market volatility increased as traders reacted to Federal Reserve policy announcements.",
            "Blue-chip stocks outperformed small caps in today's trading session.",
            "The Dow Jones Industrial Average closed at a new all-time high."
        ],
        'banking_finance': [
            "Major banks reported strong quarterly results with improved credit quality.",
            "Investment banking fees increased due to higher M&A activity.",
            "Commercial lending growth accelerated in the small business segment.",
            "Digital banking adoption continues to drive efficiency gains.",
            "Regulatory capital requirements remain well above minimum thresholds."
        ],
        'economic_indicators': [
            "Inflation data came in below expectations, easing pressure on interest rates.",
            "Unemployment claims fell to their lowest level in three months.",
            "GDP growth exceeded forecasts, indicating economic resilience.",
            "Consumer confidence index rose for the third consecutive month.",
            "Manufacturing PMI showed expansion for the first time this year."
        ],
        'federal_reserve': [
            "The Federal Reserve maintained interest rates at current levels.",
            "Fed officials signaled potential rate cuts later this year.",
            "Central bank policy remains data-dependent according to recent statements.",
            "Inflation targeting framework guides monetary policy decisions.",
            "Quantitative easing measures continue to support market liquidity."
        ],
        'general_text': [
            "The weather forecast predicts rain for the weekend.",
            "The new restaurant received excellent reviews from critics.",
            "Students are preparing for their final examinations.",
            "The movie won several awards at the film festival.",
            "Scientists discovered a new species in the rainforest."
        ]
    }
    return financial_sentences

def analyze_layer_financial_features(layer_num, llama_sae, finllama_sae, model1, model2, tokenizer1, tokenizer2, financial_sentences):
    """Analyze financial features for a specific layer"""
    print(f"\n  Analyzing Layer {layer_num}...")
    
    all_llama_activations = []
    all_finllama_activations = []
    
    # Process financial sentences
    for category, sentences in financial_sentences.items():
        for sentence in sentences:
            # Llama-2-7B model
            activations1 = get_layer_activations(sentence, model1, tokenizer1, layer_num)
            if activations1 is not None:
                encoded1 = encode_activations(activations1, llama_sae['encoder_weight'], llama_sae['encoder_bias'])
                feature_vector1 = encoded1.mean(dim=1).cpu().numpy()
                all_llama_activations.append(feature_vector1)
            
            # FinLLama-7B model
            activations2 = get_layer_activations(sentence, model2, tokenizer2, layer_num)
            if activations2 is not None:
                encoded2 = encode_activations(activations2, finllama_sae['encoder_weight'], finllama_sae['encoder_bias'])
                feature_vector2 = encoded2.mean(dim=1).cpu().numpy()
                all_finllama_activations.append(feature_vector2)
    
    # Convert to arrays
    llama_array = np.array(all_llama_activations).squeeze(1)
    finllama_array = np.array(all_finllama_activations).squeeze(1)
    
    # Calculate financial vs general activation scores
    financial_categories = ['earnings_reports', 'stock_market', 'banking_finance', 'economic_indicators', 'federal_reserve']
    general_category = 'general_text'
    
    # Get indices for financial vs general sentences
    financial_indices = []
    general_indices = []
    current_idx = 0
    
    for category, sentences in financial_sentences.items():
        if category in financial_categories:
            financial_indices.extend(range(current_idx, current_idx + len(sentences)))
        else:
            general_indices.extend(range(current_idx, current_idx + len(sentences)))
        current_idx += len(sentences)
    
    # Calculate financial vs general activation scores
    llama_financial_activation = np.mean(llama_array[financial_indices], axis=0)
    llama_general_activation = np.mean(llama_array[general_indices], axis=0)
    finllama_financial_activation = np.mean(finllama_array[financial_indices], axis=0)
    finllama_general_activation = np.mean(finllama_array[general_indices], axis=0)
    
    # Calculate financial specialization scores
    llama_financial_specialization = llama_financial_activation - llama_general_activation
    finllama_financial_specialization = finllama_financial_activation - finllama_general_activation
    
    # Calculate improvement in financial specialization
    specialization_improvement = finllama_financial_specialization - llama_financial_specialization
    
    # Find top 10 features with highest improvement
    top_10_features = np.argsort(specialization_improvement)[-10:][::-1]
    
    # Create layer results
    layer_results = []
    for rank, feature_idx in enumerate(top_10_features):
        layer_results.append({
            'layer': layer_num,
            'rank': rank + 1,
            'feature_idx': int(feature_idx),
            'llama_financial': float(llama_financial_activation[feature_idx]),
            'finllama_financial': float(finllama_financial_activation[feature_idx]),
            'llama_general': float(llama_general_activation[feature_idx]),
            'finllama_general': float(finllama_general_activation[feature_idx]),
            'llama_specialization': float(llama_financial_specialization[feature_idx]),
            'finllama_specialization': float(finllama_financial_specialization[feature_idx]),
            'specialization_improvement': float(specialization_improvement[feature_idx]),
            'improvement_percentage': float((specialization_improvement[feature_idx] / max(abs(llama_financial_specialization[feature_idx]), 1e-6)) * 100)
        })
    
    return layer_results

def analyze_financial_feature_activations():
    """Analyze which features activate most with financial words across all layers"""
    print("="*80)
    print("FINANCIAL FEATURE ACTIVATION ANALYSIS - LLAMA-2-7B vs FINLLAMA-7B (ALL LAYERS)")
    print("="*80)
    
    # Define layers to analyze
    layers = [4, 10, 16, 22, 28]
    
    # Load SAE models for all layers
    print("\n1. Loading SAE models for all layers...")
    llama_sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    finllama_sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    llama_saes = {}
    finllama_saes = {}
    
    for layer in layers:
        layer_name = f"layers.{layer}"
        try:
            llama_saes[layer] = load_sae_model(llama_sae_path, layer_name)
            finllama_saes[layer] = load_sae_model(finllama_sae_path, layer_name)
            print(f"  ✓ Loaded SAE for layer {layer}")
        except Exception as e:
            print(f"  ✗ Failed to load SAE for layer {layer}: {e}")
            continue
    
    # Load models
    print("\n2. Loading models...")
    model_path1 = "meta-llama/Llama-2-7b-hf"
    model_path2 = "cxllin/Llama2-7b-Finance"  # Corrected FinLLama model
    
    tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
    tokenizer1.pad_token = tokenizer1.eos_token
    model1 = AutoModel.from_pretrained(model_path1)
    model1.to("cuda")
    model1.eval()
    
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2, trust_remote_code=True)
    tokenizer2.pad_token = tokenizer2.eos_token
    model2 = AutoModel.from_pretrained(model_path2, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, weights_only=True, ignore_mismatched_sizes=True)
    model2.eval()
    
    # Get financial test sentences
    financial_sentences = create_financial_test_sentences()
    
    # Analyze features for each layer
    print(f"\n3. Analyzing financial features across all layers...")
    
    all_layer_results = []
    
    for layer in layers:
        if layer in llama_saes and layer in finllama_saes:
            layer_results = analyze_layer_financial_features(
                layer, llama_saes[layer], finllama_saes[layer], 
                model1, model2, tokenizer1, tokenizer2, financial_sentences
            )
            all_layer_results.extend(layer_results)
        else:
            print(f"  ✗ Skipping layer {layer} - SAE not available")
    
    # Create comprehensive results DataFrame
    df_results = pd.DataFrame(all_layer_results)
    
    # Save consolidated results
    print(f"\n4. Saving results...")
    
    # Main results file
    df_results.to_csv('3.1_financial_features_all_layers_analysis.csv', index=False)
    
    # Create summary by layer
    layer_summary = df_results.groupby('layer').agg({
        'specialization_improvement': ['mean', 'max', 'min'],
        'improvement_percentage': ['mean', 'max', 'min']
    }).round(4)
    
    layer_summary.columns = ['avg_improvement', 'max_improvement', 'min_improvement', 
                           'avg_improvement_pct', 'max_improvement_pct', 'min_improvement_pct']
    layer_summary.to_csv('3.2_layer_summary_statistics.csv')
    
    # Create visualizations
    print(f"\n5. Creating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top features by layer
    for layer in layers:
        layer_data = df_results[df_results['layer'] == layer]
        if not layer_data.empty:
            top_features = layer_data.head(5)
            ax1.scatter([layer] * len(top_features), top_features['specialization_improvement'], 
                       alpha=0.7, s=50, label=f'Layer {layer}')
    
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Specialization Improvement')
    ax1.set_title('Top 5 Financial Features by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average improvement by layer
    layer_avg = df_results.groupby('layer')['specialization_improvement'].mean()
    ax2.bar(layer_avg.index, layer_avg.values, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Average Specialization Improvement')
    ax2.set_title('Average Financial Specialization Improvement by Layer')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 10 features across all layers
    top_10_overall = df_results.nlargest(10, 'specialization_improvement')
    ax3.barh(range(len(top_10_overall)), top_10_overall['specialization_improvement'], 
             alpha=0.7, color='lightgreen')
    ax3.set_yticks(range(len(top_10_overall)))
    ax3.set_yticklabels([f'L{row["layer"]}-F{row["feature_idx"]}' for _, row in top_10_overall.iterrows()])
    ax3.set_xlabel('Specialization Improvement')
    ax3.set_title('Top 10 Financial Features Across All Layers')
    ax3.grid(True, alpha=0.3)
    
    # 4. Improvement distribution
    ax4.hist(df_results['specialization_improvement'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Specialization Improvement')
    ax4.set_ylabel('Number of Features')
    ax4.set_title('Distribution of Financial Specialization Improvements')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3.3_financial_features_all_layers_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print top results
    print(f"\n" + "="*80)
    print("TOP 10 FINANCIAL FEATURES ACROSS ALL LAYERS")
    print("="*80)
    print(f"{'Rank':<5} {'Layer':<6} {'Feature':<8} {'Improvement':<12} {'Improvement%':<12} {'Llama Fin':<10} {'FinLLama Fin':<12}")
    print("-" * 80)
    
    for _, row in df_results.nlargest(10, 'specialization_improvement').iterrows():
        print(f"{row['rank']:<5} {row['layer']:<6} {row['feature_idx']:<8} {row['specialization_improvement']:<12.4f} "
              f"{row['improvement_percentage']:<12.2f}% {row['llama_financial']:<10.4f} {row['finllama_financial']:<12.4f}")
    
    # Print layer summary
    print(f"\n" + "="*80)
    print("LAYER SUMMARY STATISTICS")
    print("="*80)
    print(f"{'Layer':<6} {'Avg Improvement':<16} {'Max Improvement':<16} {'Min Improvement':<16}")
    print("-" * 60)
    
    for layer in layers:
        if layer in layer_avg.index:
            layer_data = df_results[df_results['layer'] == layer]
            avg_imp = layer_data['specialization_improvement'].mean()
            max_imp = layer_data['specialization_improvement'].max()
            min_imp = layer_data['specialization_improvement'].min()
            print(f"{layer:<6} {avg_imp:<16.4f} {max_imp:<16.4f} {min_imp:<16.4f}")
    
    # Save summary statistics
    summary_stats = {
        'total_layers_analyzed': len(layers),
        'total_features_analyzed': len(df_results),
        'layers_with_data': list(df_results['layer'].unique()),
        'overall_avg_improvement': float(df_results['specialization_improvement'].mean()),
        'overall_max_improvement': float(df_results['specialization_improvement'].max()),
        'features_with_positive_improvement': int(len(df_results[df_results['specialization_improvement'] > 0])),
        'features_with_negative_improvement': int(len(df_results[df_results['specialization_improvement'] < 0]))
    }
    
    with open('3.4_financial_features_analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n6. Analysis complete!")
    print(f"Files generated:")
    print(f"  - 3.1_financial_features_all_layers_analysis.csv: Complete analysis of all layers")
    print(f"  - 3.2_layer_summary_statistics.csv: Layer-by-layer summary")
    print(f"  - 3.3_financial_features_all_layers_analysis.png: Visualizations")
    print(f"  - 3.4_financial_features_analysis_summary.json: Summary statistics")
    
    return {
        'df_results': df_results,
        'layer_summary': layer_summary,
        'summary_stats': summary_stats
    }

if __name__ == "__main__":
    results = analyze_financial_feature_activations()
