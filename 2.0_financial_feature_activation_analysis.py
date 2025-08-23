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

def load_sae_model(sae_path, layer_name="encoder.layer.6"):
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

def get_layer_activations(text, model, tokenizer, target_layer=6, max_length=512):
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
        encoded = torch.nn.functional.linear(activations, encoder_weight)
        
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

def analyze_financial_feature_activations():
    """Analyze which features activate most with financial words and their changes"""
    print("="*80)
    print("FINANCIAL FEATURE ACTIVATION ANALYSIS")
    print("="*80)
    
    # Load SAE models
    print("\n1. Loading SAE models...")
    bert_sae_path = "test_output/bert_layer6_k32_latents200"
    finbert_sae_path = "test_output/finbert_layer6_k32_latents200"
    
    sae1 = load_sae_model(bert_sae_path)
    sae2 = load_sae_model(finbert_sae_path)
    
    # Load models
    print("\n2. Loading models...")
    model_path1 = "bert-base-uncased"
    model_path2 = "ProsusAI/finbert"
    
    tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
    model1 = AutoModel.from_pretrained(model_path1)
    model1.to("cuda")
    model1.eval()
    
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
    model2 = AutoModel.from_pretrained(model_path2)
    model2.to("cuda")
    model2.eval()
    
    # Get financial test sentences
    financial_sentences = create_financial_test_sentences()
    
    # Analyze feature responses
    print("\n3. Analyzing feature responses...")
    
    all_bert_activations = []
    all_finbert_activations = []
    category_activations = {}
    
    for category, sentences in financial_sentences.items():
        print(f"  Processing {category} ({len(sentences)} sentences)...")
        category_bert_activations = []
        category_finbert_activations = []
        
        for i, sentence in enumerate(sentences):
            if i % 5 == 0:
                print(f"    Processing sentence {i+1}/{len(sentences)}")
            
            # BERT model
            activations1 = get_layer_activations(sentence, model1, tokenizer1)
            if activations1 is not None:
                encoded1 = encode_activations(activations1, sae1['encoder_weight'], sae1['encoder_bias'])
                feature_vector1 = encoded1.mean(dim=1).cpu().numpy()
                all_bert_activations.append(feature_vector1)
                category_bert_activations.append(feature_vector1)
            
            # FinBERT model
            activations2 = get_layer_activations(sentence, model2, tokenizer2)
            if activations2 is not None:
                encoded2 = encode_activations(activations2, sae2['encoder_weight'], sae2['encoder_bias'])
                feature_vector2 = encoded2.mean(dim=1).cpu().numpy()
                all_finbert_activations.append(feature_vector2)
                category_finbert_activations.append(feature_vector2)
        
        category_activations[category] = {
            'bert': category_bert_activations,
            'finbert': category_finbert_activations
        }
    
    # Convert to arrays
    bert_array = np.array(all_bert_activations)
    finbert_array = np.array(all_finbert_activations)
    
    print(f"BERT array shape: {bert_array.shape}")
    print(f"FinBERT array shape: {finbert_array.shape}")
    
    # Reshape to 2D (samples x features)
    bert_array = bert_array.squeeze(1)
    finbert_array = finbert_array.squeeze(1)
    
    print(f"After reshape - BERT array shape: {bert_array.shape}")
    print(f"After reshape - FinBERT array shape: {finbert_array.shape}")
    
    # Calculate feature scores
    bert_consistency = np.mean(bert_array > 0, axis=0)
    finbert_consistency = np.mean(finbert_array > 0, axis=0)
    bert_strength = np.mean(bert_array, axis=0)
    finbert_strength = np.mean(finbert_array, axis=0)
    
    bert_finetuning_score = bert_consistency * bert_strength
    finbert_finetuning_score = finbert_consistency * finbert_strength
    
    # Calculate financial-specific scores
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
    bert_financial_activation = np.mean(bert_array[financial_indices], axis=0)
    bert_general_activation = np.mean(bert_array[general_indices], axis=0)
    finbert_financial_activation = np.mean(finbert_array[financial_indices], axis=0)
    finbert_general_activation = np.mean(finbert_array[general_indices], axis=0)
    
    # Calculate financial specialization scores
    bert_financial_specialization = bert_financial_activation - bert_general_activation
    finbert_financial_specialization = finbert_financial_activation - bert_general_activation
    
    # Find top financial features
    bert_top_financial = np.argsort(bert_financial_specialization)[-30:][::-1]
    finbert_top_financial = np.argsort(finbert_financial_specialization)[-30:][::-1]
    
    print(f"\nTop 10 BERT financial features: {bert_top_financial[:10]}")
    print(f"Top 10 FinBERT financial features: {finbert_top_financial[:10]}")
    
    # Create comprehensive analysis table
    print(f"\n4. Creating comprehensive analysis table...")
    
    analysis_data = []
    
    # Analyze all 200 features
    for feature_idx in range(200):
        # BERT scores
        bert_score = bert_finetuning_score[feature_idx]
        bert_financial = bert_financial_activation[feature_idx]
        bert_general = bert_general_activation[feature_idx]
        bert_specialization = bert_financial_specialization[feature_idx]
        
        # FinBERT scores
        finbert_score = finbert_finetuning_score[feature_idx]
        finbert_financial = finbert_financial_activation[feature_idx]
        finbert_general = finbert_general_activation[feature_idx]
        finbert_specialization = finbert_financial_specialization[feature_idx]
        
        # Changes
        score_change = finbert_score - bert_score
        financial_change = finbert_financial - bert_financial
        specialization_change = finbert_specialization - bert_specialization
        
        # Percentages
        score_change_pct = (score_change / bert_score * 100) if bert_score > 0 else 0
        financial_change_pct = (financial_change / bert_financial * 100) if bert_financial > 0 else 0
        
        # Determine feature type
        if feature_idx in bert_top_financial[:20] and feature_idx in finbert_top_financial[:20]:
            feature_type = "Consistent Financial"
        elif feature_idx in bert_top_financial[:20]:
            feature_type = "BERT Financial"
        elif feature_idx in finbert_top_financial[:20]:
            feature_type = "FinBERT Financial"
        else:
            feature_type = "General"
        
        analysis_data.append({
            'feature_idx': feature_idx,
            'feature_type': feature_type,
            'bert_score': bert_score,
            'finbert_score': finbert_score,
            'score_change': score_change,
            'score_change_pct': score_change_pct,
            'bert_financial': bert_financial,
            'finbert_financial': finbert_financial,
            'financial_change': financial_change,
            'financial_change_pct': financial_change_pct,
            'bert_general': bert_general,
            'finbert_general': finbert_general_activation[feature_idx],
            'bert_specialization': bert_specialization,
            'finbert_specialization': finbert_specialization,
            'specialization_change': specialization_change
        })
    
    # Create DataFrame
    df_analysis = pd.DataFrame(analysis_data)
    
    # Sort by financial specialization change (most improved)
    df_analysis_sorted = df_analysis.sort_values('specialization_change', ascending=False)
    
    # Save comprehensive analysis
    df_analysis_sorted.to_csv('/home/nvidia/Documents/Hariom/saetrain/financial_feature_analysis_comprehensive.csv', index=False)
    
    # Create top financial features table
    print(f"\n5. Creating top financial features table...")
    
    top_financial_features = df_analysis_sorted.head(30)
    
    # Create a focused table for top features
    top_features_table = top_financial_features[['feature_idx', 'feature_type', 'bert_financial', 'finbert_financial', 
                                               'financial_change', 'financial_change_pct', 'bert_score', 'finbert_score', 
                                               'score_change', 'score_change_pct', 'specialization_change']].copy()
    
    # Round values for better readability
    for col in ['bert_financial', 'finbert_financial', 'financial_change', 'bert_score', 'finbert_score', 'score_change']:
        top_features_table[col] = top_features_table[col].round(4)
    
    for col in ['financial_change_pct', 'score_change_pct']:
        top_features_table[col] = top_features_table[col].round(2)
    
    # Save top features table
    top_features_table.to_csv('/home/nvidia/Documents/Hariom/saetrain/top_financial_features_table.csv', index=False)
    
    # Print top 20 features
    print(f"\n" + "="*80)
    print("TOP 20 FINANCIAL FEATURES - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"{'Feature':<8} {'Type':<20} {'BERT Fin':<10} {'FinBERT Fin':<12} {'Change':<10} {'Change%':<10} {'BERT Score':<12} {'FinBERT Score':<14} {'Score Change':<12} {'Score Change%':<12}")
    print("-" * 120)
    
    for _, row in top_features_table.head(20).iterrows():
        print(f"{row['feature_idx']:<8} {row['feature_type']:<20} {row['bert_financial']:<10.4f} {row['finbert_financial']:<12.4f} "
              f"{row['financial_change']:<10.4f} {row['financial_change_pct']:<10.2f}% {row['bert_score']:<12.4f} {row['finbert_score']:<14.4f} "
              f"{row['score_change']:<12.4f} {row['score_change_pct']:<12.2f}%")
    
    # Analyze emerging features
    print(f"\n" + "="*80)
    print("EMERGING FINANCIAL FEATURES ANALYSIS")
    print("="*80)
    
    emerging_features = df_analysis_sorted[df_analysis_sorted['feature_type'] == 'FinBERT Financial'].head(15)
    
    print(f"Top 15 Emerging Financial Features (New in FinBERT):")
    print(f"{'Feature':<8} {'BERT Fin':<10} {'FinBERT Fin':<12} {'Change':<10} {'Change%':<10} {'BERT Score':<12} {'FinBERT Score':<14}")
    print("-" * 80)
    
    for _, row in emerging_features.iterrows():
        print(f"{row['feature_idx']:<8} {row['bert_financial']:<10.4f} {row['finbert_financial']:<12.4f} "
              f"{row['financial_change']:<10.4f} {row['financial_change_pct']:<10.2f}% {row['bert_score']:<12.4f} {row['finbert_score']:<14.4f}")
    
    # Create visualizations
    print(f"\n6. Creating visualizations...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top financial features comparison
    top_20_features = top_features_table.head(20)
    feature_indices = top_20_features['feature_idx'].values
    bert_financial = top_20_features['bert_financial'].values
    finbert_financial = top_20_features['finbert_financial'].values
    
    x = np.arange(len(feature_indices))
    width = 0.35
    
    ax1.bar(x - width/2, bert_financial, width, label='BERT', alpha=0.7, color='skyblue')
    ax1.bar(x + width/2, finbert_financial, width, label='FinBERT', alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Financial Activation Score')
    ax1.set_title('Top 20 Financial Features Comparison')
    ax1.legend()
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([f'F{idx}' for idx in feature_indices[::2]], rotation=45)
    
    # 2. Financial specialization change
    specialization_change = top_20_features['specialization_change'].values
    colors = ['green' if change > 0 else 'red' for change in specialization_change]
    
    bars = ax2.bar(range(len(specialization_change)), specialization_change, color=colors, alpha=0.7)
    ax2.set_xlabel('Feature Rank')
    ax2.set_ylabel('Financial Specialization Change')
    ax2.set_title('Financial Specialization Improvement')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xticks(range(0, len(specialization_change), 2))
    ax2.set_xticklabels([f'F{idx}' for idx in feature_indices[::2]], rotation=45)
    
    # 3. Score change distribution
    score_changes = df_analysis_sorted['score_change'].values
    ax3.hist(score_changes, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Score Change (FinBERT - BERT)')
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Distribution of Score Changes')
    
    # 4. Financial vs General activation scatter
    ax4.scatter(df_analysis_sorted['bert_general'], df_analysis_sorted['bert_financial'], 
               alpha=0.6, label='BERT', s=20, color='skyblue')
    ax4.scatter(df_analysis_sorted['finbert_general'], df_analysis_sorted['finbert_financial'], 
               alpha=0.6, label='FinBERT', s=20, color='lightcoral')
    ax4.plot([0, max(df_analysis_sorted['bert_general'].max(), df_analysis_sorted['finbert_general'].max())], 
            [0, max(df_analysis_sorted['bert_financial'].max(), df_analysis_sorted['finbert_financial'].max())], 
            'k--', alpha=0.5, label='Equal Line')
    ax4.set_xlabel('General Text Activation')
    ax4.set_ylabel('Financial Text Activation')
    ax4.set_title('Financial vs General Activation')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('/home/nvidia/Documents/Hariom/saetrain/financial_feature_activation_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save summary statistics
    summary_stats = {
        'total_features': 200,
        'top_financial_features': len(top_financial_features),
        'emerging_financial_features': len(emerging_features),
        'avg_financial_improvement': float(df_analysis_sorted['financial_change'].mean()),
        'max_financial_improvement': float(df_analysis_sorted['financial_change'].max()),
        'features_with_positive_change': int(len(df_analysis_sorted[df_analysis_sorted['financial_change'] > 0])),
        'features_with_negative_change': int(len(df_analysis_sorted[df_analysis_sorted['financial_change'] < 0]))
    }
    
    with open('/home/nvidia/Documents/Hariom/saetrain/financial_feature_analysis_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n7. Analysis complete!")
    print(f"Files generated:")
    print(f"  - financial_feature_analysis_comprehensive.csv: Complete analysis of all 200 features")
    print(f"  - top_financial_features_table.csv: Focused table of top financial features")
    print(f"  - financial_feature_activation_analysis.png: Visualizations")
    print(f"  - financial_feature_analysis_summary.json: Summary statistics")
    
    return {
        'df_analysis': df_analysis_sorted,
        'top_features_table': top_features_table,
        'emerging_features': emerging_features,
        'summary_stats': summary_stats
    }

if __name__ == "__main__":
    results = analyze_financial_feature_activations()
