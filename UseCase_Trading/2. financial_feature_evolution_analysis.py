import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors import safe_open
import json
import os
from transformers import AutoModel, AutoTokenizer, BertModel
import pandas as pd
from collections import defaultdict
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

def load_sae_model(sae_path, layer_name):
    """Load SAE model from path for a specific layer"""
    sae_file = os.path.join(sae_path, layer_name, "sae.safetensors")
    cfg_file = os.path.join(sae_path, layer_name, "cfg.json")
    
    if not os.path.exists(sae_file) or not os.path.exists(cfg_file):
        return None
    
    # Load config
    with open(cfg_file, 'r') as f:
        sae_config = json.load(f)
    
    # Load weights
    sae_weights = {}
    with safe_open(sae_file, framework="pt", device="cuda") as f:
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
        'config': sae_config,
        'layer': layer_name
    }

def get_layer_activations_bert(text, model, tokenizer, layer_idx=0, max_length=512):
    """Get activations from specific BERT layer for given text"""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True,
        padding=True
    ).to("cuda")
    
    activations = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach())
        else:
            activations.append(output.detach())
    
    target_module = model.encoder.layer[layer_idx]
    hook = target_module.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(**inputs)
        
    hook.remove()
    
    return activations[0] if activations else None

def encode_activations(activations, encoder_weight, encoder_bias=None):
    """Encode activations using the trained SAE"""
    activations = activations.to(encoder_weight.dtype)
    
    if encoder_bias is not None:
        encoded = torch.nn.functional.linear(activations, encoder_weight, encoder_bias)
    else:
        encoded = torch.nn.functional.linear(activations, encoder_weight)
        
    encoded = torch.nn.functional.relu(encoded)
    return encoded

def create_detailed_financial_text_categories():
    """Create detailed financial text categories for granular analysis"""
    categories = {
        'market_movements': [
            "The stock market experienced significant volatility with major indices declining sharply.",
            "Equity markets rallied strongly following positive earnings announcements.",
            "Trading volume surged as investors reacted to economic data releases."
        ],
        'earnings_analysis': [
            "Quarterly earnings exceeded analyst expectations by 15% on strong revenue growth.",
            "Profit margins expanded due to operational efficiency improvements and cost controls.",
            "EBITDA margins improved to 25% from 22% in the previous quarter."
        ],
        'interest_rates_monetary': [
            "Federal Reserve maintained benchmark interest rates at current levels.",
            "Central bank signaled potential rate cuts in response to economic slowdown.",
            "Bond yields declined as investors priced in dovish monetary policy stance."
        ],
        'credit_lending': [
            "Loan approval rates tightened as banks increased risk assessment standards.",
            "Credit spreads widened reflecting heightened default risk concerns.",
            "Mortgage rates increased by 50 basis points following policy changes."
        ],
        'investment_strategies': [
            "Portfolio diversification strategies helped mitigate downside risk exposure.",
            "Hedge fund performance varied significantly based on investment approach.",
            "Asset allocation recommendations emphasized defensive positioning."
        ],
        'corporate_finance': [
            "Merger and acquisition activity accelerated with $50 billion in deals announced.",
            "Capital structure optimization reduced weighted average cost of capital.",
            "Dividend policy maintained consistent payout ratio of 40%."
        ],
        'regulatory_compliance': [
            "New financial regulations require enhanced reporting and transparency measures.",
            "Compliance costs increased due to stricter supervisory requirements.",
            "Risk management frameworks aligned with evolving regulatory expectations."
        ],
        'economic_indicators': [
            "Gross domestic product growth exceeded economist predictions for consecutive quarters.",
            "Unemployment rates declined steadily as job creation accelerated across sectors.",
            "Consumer price index inflation remained within target range of 2-3%."
        ],
        'currency_forex': [
            "Currency exchange rates fluctuated based on interest rate differentials.",
            "Foreign exchange reserves increased as central bank intervened in markets.",
            "Currency volatility spiked during geopolitical uncertainty periods."
        ],
        'commodities_trading': [
            "Oil prices surged following supply disruption concerns in key regions.",
            "Gold prices rallied as safe-haven demand increased during market stress.",
            "Agricultural commodity prices stabilized after weather-related volatility."
        ],
        'derivatives_options': [
            "Options implied volatility increased as market uncertainty rose.",
            "Derivatives trading volume surged as investors sought hedging strategies.",
            "Futures contracts pricing reflected forward-looking market expectations."
        ],
        'fintech_innovation': [
            "Digital payment adoption accelerated across retail and commercial sectors.",
            "Blockchain technology applications expanded in financial services.",
            "Artificial intelligence algorithms improved trading and risk management."
        ]
    }
    return categories

def analyze_feature_evolution_across_layers(sae_bert_path, sae_finbert_path, model_bert, model_finbert, 
                                          tokenizer_bert, tokenizer_finbert, categories):
    """Analyze how features evolve across all layers for both models"""
    print("="*80)
    print("FINANCIAL FEATURE EVOLUTION ANALYSIS")
    print("="*80)
    
    all_layers = list(range(12))  # BERT has 12 layers
    evolution_data = {
        'bert': {},
        'finbert': {},
        'improvement_scores': {},
        'emerging_features': {}
    }
    
    # Analyze each layer
    for layer_idx in all_layers:
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        # Load SAE models
        sae_bert = load_sae_model(sae_bert_path, f"encoder.layer.{layer_idx}")
        sae_finbert = load_sae_model(sae_finbert_path, f"encoder.layer.{layer_idx}")
        
        if sae_bert is None or sae_finbert is None:
            print(f"Warning: Could not load SAE models for layer {layer_idx}")
            continue
        
        # Analyze feature responses for each category
        bert_layer_data = {}
        finbert_layer_data = {}
        
        for i, (category_name, texts) in enumerate(categories.items()):
            print(f"  Processing {category_name}... ({i+1}/{len(categories)})")
            
            bert_activations = []
            finbert_activations = []
            
            for text in texts:
                # Get BERT activations
                bert_act = get_layer_activations_bert(text, model_bert, tokenizer_bert, layer_idx)
                if bert_act is not None:
                    bert_encoded = encode_activations(bert_act, sae_bert['encoder_weight'], sae_bert['encoder_bias'])
                    bert_activations.append(bert_encoded.mean(dim=1).cpu().numpy())
                
                # Get FinBERT activations
                finbert_act = get_layer_activations_bert(text, model_finbert, tokenizer_finbert, layer_idx)
                if finbert_act is not None:
                    finbert_encoded = encode_activations(finbert_act, sae_finbert['encoder_weight'], sae_finbert['encoder_bias'])
                    finbert_activations.append(finbert_encoded.mean(dim=1).cpu().numpy())
            
            if bert_activations:
                bert_activations = np.concatenate(bert_activations, axis=0)
                bert_layer_data[category_name] = bert_activations.mean(axis=0)
            
            if finbert_activations:
                finbert_activations = np.concatenate(finbert_activations, axis=0)
                finbert_layer_data[category_name] = finbert_activations.mean(axis=0)
        
        evolution_data['bert'][layer_idx] = bert_layer_data
        evolution_data['finbert'][layer_idx] = finbert_layer_data
    
    return evolution_data

def analyze_layer_wise_features(evolution_data, categories):
    """Analyze features independently for each layer"""
    print("\n" + "="*80)
    print("LAYER-WISE FEATURE ANALYSIS")
    print("="*80)
    
    num_features = 24576
    layer_analysis = {}
    
    # Analyze each layer independently
    for layer_idx in range(12):
        if layer_idx not in evolution_data['bert'] or layer_idx not in evolution_data['finbert']:
            continue
            
        print(f"\nAnalyzing Layer {layer_idx} features...")
        
        bert_data = evolution_data['bert'][layer_idx]
        finbert_data = evolution_data['finbert'][layer_idx]
        
        layer_features = []
        
        for feature_idx in range(num_features):
            if feature_idx % 5000 == 0:  # Progress update
                print(f"  Processing feature {feature_idx:,}/{num_features:,}...")
            
            # Calculate total financial activation for this feature
            bert_total = sum(bert_data.get(cat, [0]*num_features)[feature_idx] for cat in categories.keys())
            finbert_total = sum(finbert_data.get(cat, [0]*num_features)[feature_idx] for cat in categories.keys())
            
            # Calculate improvement
            improvement = finbert_total - bert_total
            improvement_percentage = (improvement / (bert_total + 1e-8)) * 100
            
            # Find top category for this feature
            top_category = None
            top_category_improvement = 0
            for category in categories.keys():
                if category in bert_data and category in finbert_data:
                    bert_cat = bert_data[category][feature_idx]
                    finbert_cat = finbert_data[category][feature_idx]
                    cat_improvement = finbert_cat - bert_cat
                    if cat_improvement > top_category_improvement:
                        top_category_improvement = cat_improvement
                        top_category = category
            
            layer_features.append({
                'feature_idx': feature_idx,
                'bert_total': bert_total,
                'finbert_total': finbert_total,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage,
                'top_category': top_category,
                'top_category_improvement': top_category_improvement
            })
        
        # Sort by improvement (descending for top, ascending for bottom)
        top_improving = sorted(layer_features, key=lambda x: x['improvement'], reverse=True)[:10]
        top_decreasing = sorted(layer_features, key=lambda x: x['improvement'])[:10]
        
        layer_analysis[layer_idx] = {
            'top_improving': top_improving,
            'top_decreasing': top_decreasing,
            'all_features': layer_features
        }
        
        print(f"  Layer {layer_idx} - Top Improving Features:")
        for i, feat in enumerate(top_improving):
            print(f"    {i+1:2d}. Feature {feat['feature_idx']:5d}: +{feat['improvement']:.3f} ({feat['improvement_percentage']:+.1f}%) - {feat['top_category']}")
        
        print(f"  Layer {layer_idx} - Top Decreasing Features:")
        for i, feat in enumerate(top_decreasing):
            print(f"    {i+1:2d}. Feature {feat['feature_idx']:5d}: {feat['improvement']:.3f} ({feat['improvement_percentage']:+.1f}%) - {feat['top_category']}")
    
    return layer_analysis

def calculate_overall_top_features(layer_analysis):
    """Calculate overall top 10 improving and decreasing features across all layers"""
    print("\n" + "="*80)
    print("OVERALL TOP FEATURES ACROSS ALL LAYERS")
    print("="*80)
    
    all_features = []
    
    # Collect all features from all layers
    for layer_idx, layer_data in layer_analysis.items():
        for feat in layer_data['all_features']:
            all_features.append({
                'layer': layer_idx,
                'feature_idx': feat['feature_idx'],
                'bert_total': feat['bert_total'],
                'finbert_total': feat['finbert_total'],
                'improvement': feat['improvement'],
                'improvement_percentage': feat['improvement_percentage'],
                'top_category': feat['top_category'],
                'top_category_improvement': feat['top_category_improvement']
            })
    
    # Sort by improvement
    top_improving_overall = sorted(all_features, key=lambda x: x['improvement'], reverse=True)[:10]
    top_decreasing_overall = sorted(all_features, key=lambda x: x['improvement'])[:10]
    
    print("\nOVERALL TOP 10 IMPROVING FEATURES:")
    print("-" * 80)
    for i, feat in enumerate(top_improving_overall):
        print(f"{i+1:2d}. Layer {feat['layer']:2d} - Feature {feat['feature_idx']:5d}: +{feat['improvement']:.3f} ({feat['improvement_percentage']:+.1f}%) - {feat['top_category']}")
    
    print("\nOVERALL TOP 10 DECREASING FEATURES:")
    print("-" * 80)
    for i, feat in enumerate(top_decreasing_overall):
        print(f"{i+1:2d}. Layer {feat['layer']:2d} - Feature {feat['feature_idx']:5d}: {feat['improvement']:.3f} ({feat['improvement_percentage']:+.1f}%) - {feat['top_category']}")
    
    return {
        'top_improving_overall': top_improving_overall,
        'top_decreasing_overall': top_decreasing_overall,
        'all_features': all_features
    }

def identify_improving_features(evolution_data, categories):
    """Identify features that consistently improve across layers"""
    print("\n" + "="*80)
    print("IDENTIFYING IMPROVING FEATURES")
    print("="*80)
    
    num_features = 24576  # Based on SAE configuration
    improvement_scores = {}
    
    # Calculate improvement scores for each feature
    for feature_idx in range(num_features):
        if feature_idx % 1000 == 0:  # Progress update every 1000 features
            print(f"    Processing feature {feature_idx:,}/{num_features:,}...")
            
        bert_scores = []
        finbert_scores = []
        
        # Collect scores across all layers
        for layer_idx in range(12):
            if layer_idx in evolution_data['bert'] and layer_idx in evolution_data['finbert']:
                # Sum across all financial categories
                bert_layer_score = sum(evolution_data['bert'][layer_idx].get(cat, [0]*num_features)[feature_idx] 
                                     for cat in categories.keys())
                finbert_layer_score = sum(evolution_data['finbert'][layer_idx].get(cat, [0]*num_features)[feature_idx] 
                                        for cat in categories.keys())
                
                bert_scores.append(bert_layer_score)
                finbert_scores.append(finbert_layer_score)
        
        if len(bert_scores) == 12 and len(finbert_scores) == 12:
            # Calculate improvement metrics
            bert_trend = np.polyfit(range(12), bert_scores, 1)[0]  # Linear trend
            finbert_trend = np.polyfit(range(12), finbert_scores, 1)[0]
            
            # Improvement score = FinBERT trend - BERT trend
            improvement_score = finbert_trend - bert_trend
            
            # Final layer advantage
            final_advantage = finbert_scores[-1] - bert_scores[-1]
            
            # Consistency score (how steadily it improves)
            finbert_improvements = [finbert_scores[i] - finbert_scores[i-1] for i in range(1, 12)]
            consistency_score = np.mean([1 if x > 0 else 0 for x in finbert_improvements])
            
            improvement_scores[feature_idx] = {
                'improvement_trend': improvement_score,
                'final_advantage': final_advantage,
                'consistency': consistency_score,
                'bert_scores': bert_scores,
                'finbert_scores': finbert_scores,
                'total_improvement': finbert_scores[-1] - bert_scores[-1]
            }
    
    return improvement_scores

def identify_emerging_features(evolution_data, categories):
    """Identify new financial features that emerge in FinBERT"""
    print("\n" + "="*80)
    print("IDENTIFYING EMERGING FEATURES")
    print("="*80)
    
    emerging_features = {}
    
    # Analyze each layer for emerging patterns
    for layer_idx in range(12):
        if layer_idx not in evolution_data['bert'] or layer_idx not in evolution_data['finbert']:
            continue
        
        bert_data = evolution_data['bert'][layer_idx]
        finbert_data = evolution_data['finbert'][layer_idx]
        
        # Find features that are strong in FinBERT but weak in BERT
        for feature_idx in range(24576):
            finbert_strength = sum(finbert_data.get(cat, [0]*24576)[feature_idx] for cat in categories.keys())
            bert_strength = sum(bert_data.get(cat, [0]*24576)[feature_idx] for cat in categories.keys())
            
            # Feature emerges if FinBERT is significantly stronger
            if finbert_strength > bert_strength * 1.5 and finbert_strength > 0.1:
                if feature_idx not in emerging_features:
                    emerging_features[feature_idx] = {
                        'layers': [],
                        'max_strength': 0,
                        'category_specialization': {}
                    }
                
                emerging_features[feature_idx]['layers'].append(layer_idx)
                emerging_features[feature_idx]['max_strength'] = max(
                    emerging_features[feature_idx]['max_strength'], finbert_strength
                )
                
                # Track category specialization
                for category in categories.keys():
                    if category in finbert_data:
                        cat_strength = finbert_data[category][feature_idx]
                        if cat_strength > 0.05:  # Significant activation
                            if category not in emerging_features[feature_idx]['category_specialization']:
                                emerging_features[feature_idx]['category_specialization'][category] = []
                            emerging_features[feature_idx]['category_specialization'][category].append({
                                'layer': layer_idx,
                                'strength': cat_strength
                            })
    
    return emerging_features

def analyze_feature_specialization(evolution_data, feature_idx, categories):
    """Analyze what a specific feature represents based on its activation patterns"""
    print(f"\nAnalyzing Feature {feature_idx} specialization...")
    
    category_strengths = {}
    
    # Analyze across all layers
    for layer_idx in range(12):
        if layer_idx in evolution_data['bert'] and layer_idx in evolution_data['finbert']:
            bert_data = evolution_data['bert'][layer_idx]
            finbert_data = evolution_data['finbert'][layer_idx]
            
            for category in categories.keys():
                if category in bert_data and category in finbert_data:
                    bert_strength = bert_data[category][feature_idx]
                    finbert_strength = finbert_data[category][feature_idx]
                    
                    if category not in category_strengths:
                        category_strengths[category] = {
                            'bert': [],
                            'finbert': [],
                            'improvement': []
                        }
                    
                    category_strengths[category]['bert'].append(bert_strength)
                    category_strengths[category]['finbert'].append(finbert_strength)
                    category_strengths[category]['improvement'].append(finbert_strength - bert_strength)
    
    # Calculate specialization metrics
    specialization_analysis = {}
    for category in category_strengths:
        bert_avg = np.mean(category_strengths[category]['bert'])
        finbert_avg = np.mean(category_strengths[category]['finbert'])
        improvement_trend = np.polyfit(range(12), category_strengths[category]['improvement'], 1)[0]
        
        specialization_analysis[category] = {
            'bert_avg': bert_avg,
            'finbert_avg': finbert_avg,
            'improvement_ratio': finbert_avg / (bert_avg + 1e-8),
            'improvement_trend': improvement_trend,
            'max_improvement': max(category_strengths[category]['improvement'])
        }
    
    return specialization_analysis

def create_evolution_visualizations(improvement_scores, emerging_features, evolution_data, categories):
    """Create comprehensive visualizations of feature evolution"""
    print("\nCreating evolution visualizations...")
    
    # Get top improving features
    top_improving = sorted(improvement_scores.items(), 
                          key=lambda x: x[1]['total_improvement'], reverse=True)[:20]
    
    # Get top emerging features
    top_emerging = sorted(emerging_features.items(), 
                         key=lambda x: x[1]['max_strength'], reverse=True)[:20]
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Top improving features evolution
    ax1 = plt.subplot(4, 4, 1)
    for i, (feature_idx, data) in enumerate(top_improving[:5]):
        ax1.plot(range(12), data['finbert_scores'], 
                label=f'FinBERT Feature {feature_idx}', linewidth=2)
        ax1.plot(range(12), data['bert_scores'], 
                label=f'BERT Feature {feature_idx}', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Feature Activation')
    ax1.set_title('Top 5 Improving Features Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement distribution
    ax2 = plt.subplot(4, 4, 2)
    improvements = [data['total_improvement'] for _, data in top_improving]
    ax2.hist(improvements, bins=20, alpha=0.7, color='green')
    ax2.set_xlabel('Total Improvement (FinBERT - BERT)')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Distribution of Feature Improvements')
    
    # 3. Layer-wise improvement heatmap
    ax3 = plt.subplot(4, 4, 3)
    improvement_matrix = np.zeros((20, 12))
    for i, (feature_idx, data) in enumerate(top_improving):
        for layer in range(12):
            improvement_matrix[i, layer] = data['finbert_scores'][layer] - data['bert_scores'][layer]
    
    im3 = ax3.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Feature Rank')
    ax3.set_title('Top 20 Features: Layer-wise Improvement')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Emerging features strength
    ax4 = plt.subplot(4, 4, 4)
    emerging_strengths = [data['max_strength'] for _, data in top_emerging]
    emerging_indices = [idx for idx, _ in top_emerging]
    ax4.bar(range(len(emerging_strengths)), emerging_strengths, alpha=0.7, color='orange')
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Maximum Strength')
    ax4.set_title('Top Emerging Features Strength')
    
    # 5-8. Category specialization for top features
    for i, (feature_idx, data) in enumerate(top_improving[:4]):
        ax = plt.subplot(4, 4, 5 + i)
        
        # Get category strengths for this feature
        category_strengths = []
        category_names = []
        for category in categories.keys():
            bert_avg = np.mean([evolution_data['bert'][layer].get(category, [0]*24576)[feature_idx] 
                              for layer in range(12) if layer in evolution_data['bert']])
            finbert_avg = np.mean([evolution_data['finbert'][layer].get(category, [0]*24576)[feature_idx] 
                                 for layer in range(12) if layer in evolution_data['finbert']])
            
            category_strengths.append([bert_avg, finbert_avg])
            category_names.append(category.replace('_', '\n'))
        
        category_strengths = np.array(category_strengths)
        
        x = np.arange(len(category_names))
        width = 0.35
        ax.bar(x - width/2, category_strengths[:, 0], width, label='BERT', alpha=0.7)
        ax.bar(x + width/2, category_strengths[:, 1], width, label='FinBERT', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(category_names, rotation=45, ha='right')
        ax.set_ylabel('Average Activation')
        ax.set_title(f'Feature {feature_idx}: Category Specialization')
        ax.legend()
    
    # 9-12. Emerging features category analysis
    for i, (feature_idx, data) in enumerate(top_emerging[:4]):
        ax = plt.subplot(4, 4, 9 + i)
        
        # Analyze category specialization for emerging feature
        categories_analyzed = list(data['category_specialization'].keys())
        max_strengths = [max([item['strength'] for item in data['category_specialization'][cat]]) 
                        for cat in categories_analyzed]
        
        ax.bar(range(len(max_strengths)), max_strengths, alpha=0.7, color='red')
        ax.set_xticks(range(len(categories_analyzed)))
        ax.set_xticklabels([cat.replace('_', '\n') for cat in categories_analyzed], 
                          rotation=45, ha='right')
        ax.set_ylabel('Maximum Strength')
        ax.set_title(f'Emerging Feature {feature_idx}: Category Strengths')
    
    # 13. Overall improvement trends
    ax13 = plt.subplot(4, 4, 13)
    layer_improvements = []
    for layer in range(12):
        layer_total = 0
        count = 0
        for _, data in improvement_scores.items():
            if len(data['finbert_scores']) > layer:
                layer_total += data['finbert_scores'][layer] - data['bert_scores'][layer]
                count += 1
        layer_improvements.append(layer_total / count if count > 0 else 0)
    
    ax13.plot(range(12), layer_improvements, marker='o', linewidth=2, markersize=8)
    ax13.set_xlabel('Layer')
    ax13.set_ylabel('Average Improvement')
    ax13.set_title('Layer-wise Average Improvement')
    ax13.grid(True, alpha=0.3)
    
    # 14. Consistency analysis
    ax14 = plt.subplot(4, 4, 14)
    consistencies = [data['consistency'] for data in improvement_scores.values()]
    ax14.hist(consistencies, bins=20, alpha=0.7, color='purple')
    ax14.set_xlabel('Consistency Score (0-1)')
    ax14.set_ylabel('Number of Features')
    ax14.set_title('Feature Improvement Consistency')
    
    # 15. Final layer advantage distribution
    ax15 = plt.subplot(4, 4, 15)
    final_advantages = [data['final_advantage'] for data in improvement_scores.values()]
    ax15.hist(final_advantages, bins=30, alpha=0.7, color='blue')
    ax15.set_xlabel('Final Layer Advantage')
    ax15.set_ylabel('Number of Features')
    ax15.set_title('Final Layer Advantage Distribution')
    
    # 16. Summary statistics
    ax16 = plt.subplot(4, 4, 16)
    ax16.axis('off')
    
    total_features = len(improvement_scores)
    improving_features = sum(1 for data in improvement_scores.values() if data['total_improvement'] > 0)
    emerging_count = len(emerging_features)
    
    summary_text = f"""
    FEATURE EVOLUTION SUMMARY
    
    Total Features Analyzed: {total_features:,}
    Improving Features: {improving_features:,} ({improving_features/total_features*100:.1f}%)
    Emerging Features: {emerging_count:,}
    
    Top Improvements:
    - Max Improvement: {max([data['total_improvement'] for data in improvement_scores.values()]):.3f}
    - Avg Improvement: {np.mean([data['total_improvement'] for data in improvement_scores.values()]):.3f}
    
    Consistency:
    - Avg Consistency: {np.mean([data['consistency'] for data in improvement_scores.values()]):.2f}
    """
    
    ax16.text(0.1, 0.9, summary_text, transform=ax16.transAxes, fontsize=10, 
              verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/financial_feature_evolution.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_layer_wise_reports(layer_analysis, overall_analysis):
    """Generate detailed reports for layer-wise analysis"""
    print("\n" + "="*80)
    print("GENERATING LAYER-WISE REPORTS")
    print("="*80)
    
    # Create layer-wise report
    layer_report_data = []
    
    for layer_idx, layer_data in layer_analysis.items():
        # Top improving features for this layer
        for i, feat in enumerate(layer_data['top_improving']):
            layer_report_data.append({
                'layer': layer_idx,
                'feature_idx': feat['feature_idx'],
                'rank': i + 1,
                'type': 'improving',
                'bert_total': feat['bert_total'],
                'finbert_total': feat['finbert_total'],
                'improvement': feat['improvement'],
                'improvement_percentage': feat['improvement_percentage'],
                'top_category': feat['top_category'],
                'top_category_improvement': feat['top_category_improvement']
            })
        
        # Top decreasing features for this layer
        for i, feat in enumerate(layer_data['top_decreasing']):
            layer_report_data.append({
                'layer': layer_idx,
                'feature_idx': feat['feature_idx'],
                'rank': i + 1,
                'type': 'decreasing',
                'bert_total': feat['bert_total'],
                'finbert_total': feat['finbert_total'],
                'improvement': feat['improvement'],
                'improvement_percentage': feat['improvement_percentage'],
                'top_category': feat['top_category'],
                'top_category_improvement': feat['top_category_improvement']
            })
    
    # Create overall report
    overall_report_data = []
    
    # Top improving overall
    for i, feat in enumerate(overall_analysis['top_improving_overall']):
        overall_report_data.append({
            'rank': i + 1,
            'type': 'improving',
            'layer': feat['layer'],
            'feature_idx': feat['feature_idx'],
            'bert_total': feat['bert_total'],
            'finbert_total': feat['finbert_total'],
            'improvement': feat['improvement'],
            'improvement_percentage': feat['improvement_percentage'],
            'top_category': feat['top_category'],
            'top_category_improvement': feat['top_category_improvement']
        })
    
    # Top decreasing overall
    for i, feat in enumerate(overall_analysis['top_decreasing_overall']):
        overall_report_data.append({
            'rank': i + 1,
            'type': 'decreasing',
            'layer': feat['layer'],
            'feature_idx': feat['feature_idx'],
            'bert_total': feat['bert_total'],
            'finbert_total': feat['finbert_total'],
            'improvement': feat['improvement'],
            'improvement_percentage': feat['improvement_percentage'],
            'top_category': feat['top_category'],
            'top_category_improvement': feat['top_category_improvement']
        })
    
    # Save reports
    df_layer_report = pd.DataFrame(layer_report_data)
    df_overall_report = pd.DataFrame(overall_report_data)
    
    df_layer_report.to_csv('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/layer_wise_feature_report.csv', index=False)
    df_overall_report.to_csv('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/overall_top_features_report.csv', index=False)
    
    print(f"Layer-wise report saved: layer_wise_feature_report.csv")
    print(f"Overall report saved: overall_top_features_report.csv")
    
    return df_layer_report, df_overall_report

def create_layer_wise_visualizations(layer_analysis, overall_analysis):
    """Create visualizations for layer-wise analysis"""
    print("\nCreating layer-wise visualizations...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Plot top improving features for each layer
    for layer_idx in range(12):
        if layer_idx in layer_analysis:
            ax = axes[layer_idx]
            
            # Get top 5 improving features for this layer
            top_improving = layer_analysis[layer_idx]['top_improving'][:5]
            
            feature_indices = [f"F{feat['feature_idx']}" for feat in top_improving]
            improvements = [feat['improvement'] for feat in top_improving]
            percentages = [feat['improvement_percentage'] for feat in top_improving]
            
            bars = ax.bar(range(len(improvements)), improvements, alpha=0.7, color='green')
            ax.set_xticks(range(len(feature_indices)))
            ax.set_xticklabels(feature_indices, rotation=45, ha='right')
            ax.set_ylabel('Improvement')
            ax.set_title(f'Layer {layer_idx} - Top 5 Improving Features')
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{pct:+.0f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/layer_wise_improvements.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create overall comparison plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Overall top improving features
    top_improving_overall = overall_analysis['top_improving_overall'][:10]
    feature_labels = [f"L{feat['layer']}-F{feat['feature_idx']}" for feat in top_improving_overall]
    improvements = [feat['improvement'] for feat in top_improving_overall]
    percentages = [feat['improvement_percentage'] for feat in top_improving_overall]
    
    bars1 = ax1.bar(range(len(improvements)), improvements, alpha=0.7, color='green')
    ax1.set_xticks(range(len(feature_labels)))
    ax1.set_xticklabels(feature_labels, rotation=45, ha='right')
    ax1.set_ylabel('Improvement')
    ax1.set_title('Overall Top 10 Improving Features')
    ax1.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars1, percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{pct:+.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Overall top decreasing features
    top_decreasing_overall = overall_analysis['top_decreasing_overall'][:10]
    feature_labels = [f"L{feat['layer']}-F{feat['feature_idx']}" for feat in top_decreasing_overall]
    improvements = [feat['improvement'] for feat in top_decreasing_overall]
    percentages = [feat['improvement_percentage'] for feat in top_decreasing_overall]
    
    bars2 = ax2.bar(range(len(improvements)), improvements, alpha=0.7, color='red')
    ax2.set_xticks(range(len(feature_labels)))
    ax2.set_xticklabels(feature_labels, rotation=45, ha='right')
    ax2.set_ylabel('Improvement')
    ax2.set_title('Overall Top 10 Decreasing Features')
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars2, percentages)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height - 0.01,
               f'{pct:+.0f}%', ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/overall_feature_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_feature_report(improvement_scores, emerging_features, evolution_data, categories):
    """Generate detailed report of top improving and emerging features"""
    print("\n" + "="*80)
    print("GENERATING DETAILED FEATURE REPORT")
    print("="*80)
    
    # Get top 20 improving features
    top_improving = sorted(improvement_scores.items(), 
                          key=lambda x: x[1]['total_improvement'], reverse=True)[:20]
    
    # Get top 20 emerging features
    top_emerging = sorted(emerging_features.items(), 
                         key=lambda x: x[1]['max_strength'], reverse=True)[:20]
    
    report_data = []
    
    print("\nTOP 20 IMPROVING FEATURES:")
    print("-" * 100)
    
    for i, (feature_idx, data) in enumerate(top_improving):
        # Analyze specialization
        specialization = analyze_feature_specialization(evolution_data, feature_idx, categories)
        
        # Find top categories
        top_categories = sorted(specialization.items(), 
                              key=lambda x: x[1]['improvement_ratio'], reverse=True)[:3]
        
        print(f"{i+1:2d}. Feature {feature_idx:5d}:")
        print(f"    Total Improvement: {data['total_improvement']:.3f}")
        print(f"    Consistency: {data['consistency']:.2f}")
        print(f"    Top Categories: {[cat.replace('_', ' ').title() for cat, _ in top_categories]}")
        print(f"    Improvement Trend: {data['improvement_trend']:.4f}")
        print()
        
        # Store for CSV
        report_data.append({
            'feature_idx': feature_idx,
            'feature_type': 'improving',
            'rank': i + 1,
            'total_improvement': data['total_improvement'],
            'consistency': data['consistency'],
            'improvement_trend': data['improvement_trend'],
            'final_advantage': data['final_advantage'],
            'top_category_1': top_categories[0][0] if top_categories else 'None',
            'top_category_2': top_categories[1][0] if len(top_categories) > 1 else 'None',
            'top_category_3': top_categories[2][0] if len(top_categories) > 2 else 'None',
            'category_1_improvement': top_categories[0][1]['improvement_ratio'] if top_categories else 0,
            'category_2_improvement': top_categories[1][1]['improvement_ratio'] if len(top_categories) > 1 else 0,
            'category_3_improvement': top_categories[2][1]['improvement_ratio'] if len(top_categories) > 2 else 0
        })
    
    print("\nTOP 20 EMERGING FEATURES:")
    print("-" * 100)
    
    for i, (feature_idx, data) in enumerate(top_emerging):
        print(f"{i+1:2d}. Feature {feature_idx:5d}:")
        print(f"    Max Strength: {data['max_strength']:.3f}")
        print(f"    Emerges in Layers: {data['layers']}")
        print(f"    Specialized Categories: {list(data['category_specialization'].keys())}")
        print()
        
        # Store for CSV
        report_data.append({
            'feature_idx': feature_idx,
            'feature_type': 'emerging',
            'rank': i + 1,
            'max_strength': data['max_strength'],
            'emerging_layers': str(data['layers']),
            'specialized_categories': str(list(data['category_specialization'].keys())),
            'total_improvement': 0,  # Not applicable for emerging features
            'consistency': 0,
            'improvement_trend': 0,
            'final_advantage': 0,
            'top_category_1': 'None',
            'top_category_2': 'None',
            'top_category_3': 'None',
            'category_1_improvement': 0,
            'category_2_improvement': 0,
            'category_3_improvement': 0
        })
    
    # Save detailed report
    df_report = pd.DataFrame(report_data)
    df_report.to_csv('/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/feature_evolution_detailed_report.csv', 
                     index=False)
    
    return df_report

def main():
    """Main analysis function"""
    print("="*80)
    print("FINANCIAL FEATURE EVOLUTION ANALYSIS")
    print("="*80)
    
    # Set paths
    bert_sae_path = "/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/SAE_AllLayers_BERT/unnamed"
    finbert_sae_path = "/home/nvidia/Documents/Hariom/SAELens/FinBertAnalysis/SAE_AllLayers_Finbert/unnamed"
    
    # Load models
    print("Loading base models...")
    tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased').to("cuda")
    
    tokenizer_finbert = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model_finbert = BertModel.from_pretrained('ProsusAI/finbert').to("cuda")
    
    # Create detailed financial categories
    categories = create_detailed_financial_text_categories()
    
    # Analyze feature evolution
    evolution_data = analyze_feature_evolution_across_layers(
        bert_sae_path, finbert_sae_path, model_bert, model_finbert, 
        tokenizer_bert, tokenizer_finbert, categories
    )
    
    # Perform layer-wise analysis
    layer_analysis = analyze_layer_wise_features(evolution_data, categories)
    
    # Calculate overall top features
    overall_analysis = calculate_overall_top_features(layer_analysis)
    
    # Generate layer-wise reports and visualizations
    df_layer_report, df_overall_report = generate_layer_wise_reports(layer_analysis, overall_analysis)
    create_layer_wise_visualizations(layer_analysis, overall_analysis)
    
    # Original analysis (optional - can be commented out to save time)
    print("\n" + "="*80)
    print("ORIGINAL EVOLUTION ANALYSIS (Optional)")
    print("="*80)
    
    # Identify improving features
    improvement_scores = identify_improving_features(evolution_data, categories)
    
    # Identify emerging features
    emerging_features = identify_emerging_features(evolution_data, categories)
    
    # Create visualizations
    create_evolution_visualizations(improvement_scores, emerging_features, evolution_data, categories)
    
    # Generate detailed report
    df_report = generate_detailed_feature_report(improvement_scores, emerging_features, evolution_data, categories)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Files generated:")
    print(f"- Layer-wise improvements: layer_wise_improvements.png")
    print(f"- Overall feature comparison: overall_feature_comparison.png")
    print(f"- Layer-wise report: layer_wise_feature_report.csv")
    print(f"- Overall top features: overall_top_features_report.csv")
    print(f"- Feature evolution visualization: financial_feature_evolution.png")
    print(f"- Detailed evolution report: feature_evolution_detailed_report.csv")
    
    return {
        'evolution_data': evolution_data,
        'layer_analysis': layer_analysis,
        'overall_analysis': overall_analysis,
        'improvement_scores': improvement_scores,
        'emerging_features': emerging_features,
        'layer_report_df': df_layer_report,
        'overall_report_df': df_overall_report,
        'evolution_report_df': df_report
    }

if __name__ == "__main__":
    results = main() 