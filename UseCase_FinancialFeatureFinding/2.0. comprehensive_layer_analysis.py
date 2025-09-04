#!/usr/bin/env python3
"""
Comprehensive SAE Layer Analysis Script
Analyzes feature drift across all layers (4, 10, 16, 22, 28) for Llama-2-7B vs FinLLama-7B
Generates detailed reports, visualizations, and summary tables
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from safetensors import safe_open
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ComprehensiveSAEAnalyzer:
    def __init__(self):
        self.layers = [4, 10, 16, 22, 28]
        self.base_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models"
        self.results = {}
        self.feature_categories = {}
        
    def load_sae_model(self, path):
        """Load SAE model from path"""
        try:
            config_path = os.path.join(path, 'cfg.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            sae_path = os.path.join(path, 'sae.safetensors')
            with safe_open(sae_path, framework='pt', device='cpu') as f:
                encoder_weight = f.get_tensor('encoder.weight')
                decoder_weight = f.get_tensor('W_dec')
            
            return {
                'encoder_weight': encoder_weight,
                'decoder_weight': decoder_weight,
                'config': config
            }
        except Exception as e:
            print(f"❌ Error loading model from {path}: {e}")
            return None
    
    def analyze_layer(self, layer_num):
        """Analyze a specific layer"""
        print(f"\n🔍 ANALYZING LAYER {layer_num}")
        print("=" * 60)
        
        # Define paths
        sae_path1 = f"{self.base_path}/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun/layers.{layer_num}"
        sae_path2 = f"{self.base_path}/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun/layers.{layer_num}"
        
        # Load models
        sae1 = self.load_sae_model(sae_path1)
        sae2 = self.load_sae_model(sae_path2)
        
        if sae1 is None or sae2 is None:
            print(f"❌ Failed to load models for layer {layer_num}")
            return None
        
        # Extract weights
        encoder1 = sae1['encoder_weight'].numpy()
        encoder2 = sae2['encoder_weight'].numpy()
        decoder1 = sae1['decoder_weight'].numpy()
        decoder2 = sae2['decoder_weight'].numpy()
        
        print(f"✅ Models loaded - Encoder: {encoder1.shape}, Decoder: {decoder1.shape}")
        
        # Calculate correlations
        encoder_corr = np.corrcoef(encoder1.flatten(), encoder2.flatten())[0, 1]
        decoder_corr = np.corrcoef(decoder1.flatten(), decoder2.flatten())[0, 1]
        
        # Feature-wise drift analysis
        feature_drifts = []
        feature_correlations = []
        
        for i in range(encoder1.shape[0]):
            # Normalize vectors for fair comparison
            enc1_norm = encoder1[i] / np.linalg.norm(encoder1[i])
            enc2_norm = encoder2[i] / np.linalg.norm(encoder2[i])
            
            # Calculate drift
            drift = np.linalg.norm(enc1_norm - enc2_norm)
            feature_drifts.append(drift)
            
            # Calculate correlation for this feature
            corr = np.corrcoef(enc1_norm, enc2_norm)[0, 1]
            feature_correlations.append(corr)
        
        feature_drifts = np.array(feature_drifts)
        feature_correlations = np.array(feature_correlations)
        
        # Categorize features
        stable_features = np.where(feature_drifts < 0.5)[0]
        moderate_features = np.where((feature_drifts >= 0.5) & (feature_drifts < 1.0))[0]
        high_drift_features = np.where(feature_drifts >= 1.0)[0]
        
        # Find top drifted and least drifted
        top_drifted = np.argsort(feature_drifts)[-20:]
        least_drifted = np.argsort(feature_drifts)[:20]
        
        # Store results
        layer_results = {
            'layer': layer_num,
            'encoder_correlation': float(encoder_corr),
            'decoder_correlation': float(decoder_corr),
            'mean_drift': float(np.mean(feature_drifts)),
            'std_drift': float(np.std(feature_drifts)),
            'min_drift': float(np.min(feature_drifts)),
            'max_drift': float(np.max(feature_drifts)),
            'feature_drifts': feature_drifts.tolist(),
            'feature_correlations': feature_correlations.tolist(),
            'stable_features': stable_features.tolist(),
            'moderate_features': moderate_features.tolist(),
            'high_drift_features': high_drift_features.tolist(),
            'top_drifted_features': top_drifted.tolist(),
            'least_drifted_features': least_drifted.tolist(),
            'feature_counts': {
                'stable': int(len(stable_features)),
                'moderate': int(len(moderate_features)),
                'high_drift': int(len(high_drift_features))
            }
        }
        
        self.results[layer_num] = layer_results
        
        # Print summary
        print(f"📊 LAYER {layer_num} RESULTS:")
        print(f"   Encoder Correlation: {encoder_corr:.4f}")
        print(f"   Decoder Correlation: {decoder_corr:.4f}")
        print(f"   Mean Drift: {np.mean(feature_drifts):.4f}")
        print(f"   Feature Distribution:")
        print(f"     Stable: {len(stable_features)} ({len(stable_features)/len(feature_drifts)*100:.1f}%)")
        print(f"     Moderate: {len(moderate_features)} ({len(moderate_features)/len(feature_drifts)*100:.1f}%)")
        print(f"     High Drift: {len(high_drift_features)} ({len(high_drift_features)/len(feature_drifts)*100:.1f}%)")
        
        return layer_results
    
    def create_layer_comparison_visualization(self):
        """Create comprehensive visualization comparing all layers"""
        print("\n🎨 Creating comprehensive layer comparison visualization...")
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('Comprehensive SAE Layer Analysis: Llama-2-7B vs FinLLama-7B', fontsize=16, fontweight='bold')
        
        # 1. Overall correlations across layers
        layers = list(self.results.keys())
        encoder_corrs = [self.results[l]['encoder_correlation'] for l in layers]
        decoder_corrs = [self.results[l]['decoder_correlation'] for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, encoder_corrs, width, label='Encoder', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, decoder_corrs, width, label='Decoder', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('Layer Number')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].set_title('Weight Correlations Across Layers')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(layers)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drift statistics across layers
        mean_drifts = [self.results[l]['mean_drift'] for l in layers]
        std_drifts = [self.results[l]['std_drift'] for l in layers]
        
        axes[0, 1].bar(layers, mean_drifts, alpha=0.7, color='orange', yerr=std_drifts, capsize=5)
        axes[0, 1].set_xlabel('Layer Number')
        axes[0, 1].set_ylabel('Mean Drift')
        axes[0, 1].set_title('Feature Drift Across Layers')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature distribution across layers
        stable_counts = [self.results[l]['feature_counts']['stable'] for l in layers]
        moderate_counts = [self.results[l]['feature_counts']['moderate'] for l in layers]
        high_drift_counts = [self.results[l]['feature_counts']['high_drift'] for l in layers]
        
        bottom = np.zeros(len(layers))
        axes[0, 2].bar(layers, stable_counts, label='Stable', bottom=bottom, alpha=0.8, color='green')
        bottom += stable_counts
        axes[0, 2].bar(layers, moderate_counts, label='Moderate', bottom=bottom, alpha=0.8, color='yellow')
        bottom += moderate_counts
        axes[0, 2].bar(layers, high_drift_counts, label='High Drift', bottom=bottom, alpha=0.8, color='red')
        axes[0, 2].set_xlabel('Layer Number')
        axes[0, 2].set_ylabel('Number of Features')
        axes[0, 2].set_title('Feature Distribution by Drift Category')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Drift distribution comparison
        for i, layer in enumerate(layers):
            row = (i + 1) // 3
            col = (i + 1) % 3
            if row < 3 and col < 4:
                feature_drifts = np.array(self.results[layer]['feature_drifts'])
                axes[row, col].hist(feature_drifts, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
                axes[row, col].set_xlabel('Drift Score')
                axes[row, col].set_ylabel('Count')
                axes[row, col].set_title(f'Layer {layer} Drift Distribution')
                axes[row, col].axvline(np.mean(feature_drifts), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(feature_drifts):.3f}')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_layer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualization saved as 'comprehensive_layer_analysis.png'")
    
    def create_summary_table(self):
        """Create summary table for all layers"""
        print("\n📊 Creating summary table...")
        
        # Prepare data for table
        table_data = []
        for layer in self.layers:
            if layer in self.results:
                result = self.results[layer]
                table_data.append({
                    'Layer': layer,
                    'Encoder Corr': f"{result['encoder_correlation']:.4f}",
                    'Decoder Corr': f"{result['decoder_correlation']:.4f}",
                    'Mean Drift': f"{result['mean_drift']:.4f}",
                    'Std Drift': f"{result['std_drift']:.4f}",
                    'Stable Features': result['feature_counts']['stable'],
                    'Moderate Features': result['feature_counts']['moderate'],
                    'High Drift Features': result['feature_counts']['high_drift'],
                    'Stable %': f"{result['feature_counts']['stable']/400*100:.1f}%",
                    'Moderate %': f"{result['feature_counts']['moderate']/400*100:.1f}%",
                    'High Drift %': f"{result['feature_counts']['high_drift']/400*100:.1f}%"
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        df.to_csv('layer_analysis_summary.csv', index=False)
        df.to_html('layer_analysis_summary.html', index=False)
        
        print("📋 Summary table saved as:")
        print("   - layer_analysis_summary.csv")
        print("   - layer_analysis_summary.html")
        
        return df
    
    def create_feature_cross_layer_analysis(self):
        """Analyze feature behavior across all layers"""
        print("\n🔍 Creating cross-layer feature analysis...")
        
        # Find features that are consistently stable or high drift across layers
        all_features = set(range(400))
        consistently_stable = all_features.copy()
        consistently_high_drift = all_features.copy()
        
        for layer in self.layers:
            if layer in self.results:
                stable_features = set(self.results[layer]['stable_features'])
                high_drift_features = set(self.results[layer]['high_drift_features'])
                
                consistently_stable &= stable_features
                consistently_high_drift &= high_drift_features
        
        # Create cross-layer analysis
        cross_layer_analysis = {
            'consistently_stable_features': sorted(list(consistently_stable)),
            'consistently_high_drift_features': sorted(list(consistently_high_drift)),
            'consistently_stable_count': len(consistently_stable),
            'consistently_high_drift_count': len(consistently_high_drift),
            'layer_wise_stability': {}
        }
        
        # Analyze each feature's behavior across layers
        for feature_idx in range(400):
            feature_stability = []
            for layer in self.layers:
                if layer in self.results:
                    drift = self.results[layer]['feature_drifts'][feature_idx]
                    if drift < 0.5:
                        feature_stability.append('stable')
                    elif drift < 1.0:
                        feature_stability.append('moderate')
                    else:
                        feature_stability.append('high_drift')
            
            cross_layer_analysis['layer_wise_stability'][feature_idx] = feature_stability
        
        # Save cross-layer analysis
        with open('cross_layer_feature_analysis.json', 'w') as f:
            json.dump(cross_layer_analysis, f, indent=2)
        
        print("🔗 Cross-layer feature analysis saved as 'cross_layer_feature_analysis.json'")
        print(f"📊 Consistently stable features: {len(consistently_stable)}")
        print(f"📊 Consistently high drift features: {len(consistently_high_drift)}")
        
        return cross_layer_analysis
    
    def generate_readme(self):
        """Generate comprehensive README file"""
        print("\n📝 Generating comprehensive README...")
        
        readme_content = f"""# Comprehensive SAE Layer Analysis Report

## Overview
This report contains a comprehensive analysis of Sparse Autoencoder (SAE) models across 5 layers (4, 10, 16, 22, 28) comparing Llama-2-7B base model vs FinLLama-7B fine-tuned model.

## Analysis Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Layers Analyzed
- Layer 4
- Layer 10  
- Layer 16
- Layer 22
- Layer 28

## Key Findings

### Overall Model Similarity
"""
        
        # Add overall statistics
        total_encoder_corr = np.mean([self.results[l]['encoder_correlation'] for l in self.layers if l in self.results])
        total_decoder_corr = np.mean([self.results[l]['decoder_correlation'] for l in self.layers if l in self.results])
        
        readme_content += f"""
- **Average Encoder Correlation**: {total_encoder_corr:.4f} ({total_encoder_corr*100:.1f}%)
- **Average Decoder Correlation**: {total_decoder_corr:.4f} ({total_decoder_corr*100:.1f}%)
- **Overall Model Similarity**: {((total_encoder_corr + total_decoder_corr)/2)*100:.1f}%

### Feature Drift Summary
"""
        
        # Add feature drift summary
        total_stable = sum([self.results[l]['feature_counts']['stable'] for l in self.layers if l in self.results])
        total_moderate = sum([self.results[l]['feature_counts']['moderate'] for l in self.layers if l in self.results])
        total_high_drift = sum([self.results[l]['feature_counts']['high_drift'] for l in self.layers if l in self.results])
        
        readme_content += f"""
- **Total Stable Features**: {total_stable} ({(total_stable/(len(self.layers)*400))*100:.1f}%)
- **Total Moderate Drift Features**: {total_moderate} ({(total_moderate/(len(self.layers)*400))*100:.1f}%)
- **Total High Drift Features**: {total_high_drift} ({(total_high_drift/(len(self.layers)*400))*100:.1f}%)

## Layer-by-Layer Analysis
"""
        
        # Add layer-by-layer analysis
        for layer in self.layers:
            if layer in self.results:
                result = self.results[layer]
                readme_content += f"""
### Layer {layer}
- **Encoder Correlation**: {result['encoder_correlation']:.4f}
- **Decoder Correlation**: {result['decoder_correlation']:.4f}
- **Mean Drift**: {result['mean_drift']:.4f}
- **Feature Distribution**:
  - Stable: {result['feature_counts']['stable']} ({result['feature_counts']['stable']/400*100:.1f}%)
  - Moderate: {result['feature_counts']['moderate']} ({result['feature_counts']['moderate']/400*100:.1f}%)
  - High Drift: {result['feature_counts']['high_drift']} ({result['feature_counts']['high_drift']/400*100:.1f}%)

"""
        
        readme_content += """
## Files Generated

### Data Files
- `comprehensive_layer_analysis.json` - Complete analysis results
- `layer_analysis_summary.csv` - Summary table in CSV format
- `layer_analysis_summary.html` - Summary table in HTML format
- `cross_layer_feature_analysis.json` - Cross-layer feature behavior analysis

### Visualizations
- `comprehensive_layer_analysis.png` - Comprehensive visualization of all layers
- Individual layer visualizations (if generated)

## Interpretation

### What the Results Mean
1. **Low Correlations**: Indicate significant fine-tuning with financial data
2. **High Feature Drift**: Shows major reorganization of feature representations
3. **Layer Differences**: Different layers may specialize in different aspects of language

### Feature Categories
- **Stable Features (< 0.5 drift)**: Preserved knowledge from base model
- **Moderate Drift (0.5-1.0)**: Partially modified features
- **High Drift (≥ 1.0)**: Completely reorganized features for financial domain

## Usage

### Running the Analysis
```bash
python comprehensive_layer_analysis.py
```

### Accessing Results
- Check the generated CSV/HTML files for tabular data
- View the PNG file for visualizations
- Use the JSON files for programmatic access to results

## Technical Details

### Model Configuration
- Base Model: meta-llama/Llama-2-7b-hf
- Fine-tuned Model: cxllin/Llama2-7b-Finance
- SAE Configuration: 400 latents, k=32, expansion_factor=32
- Training Data: WikiText-103

### Analysis Methodology
- Feature drift calculated using normalized vector distances
- Correlations computed on flattened weight matrices
- Feature categorization based on drift thresholds
- Cross-layer analysis for consistent behavior patterns

---
*Generated by Comprehensive SAE Layer Analyzer*
"""
        
        # Save README
        with open('COMPREHENSIVE_ANALYSIS_README.md', 'w') as f:
            f.write(readme_content)
        
        print("📖 README saved as 'COMPREHENSIVE_ANALYSIS_README.md'")
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE SAE LAYER ANALYSIS")
        print("=" * 80)
        print(f"Analyzing layers: {self.layers}")
        print("=" * 80)
        
        # Analyze each layer
        for layer in self.layers:
            self.analyze_layer(layer)
        
        # Generate comprehensive results
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE RESULTS")
        print("="*80)
        
        # Save complete results
        with open('comprehensive_layer_analysis.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("💾 Complete analysis results saved as 'comprehensive_layer_analysis.json'")
        
        # Create visualizations and tables
        self.create_layer_comparison_visualization()
        summary_df = self.create_summary_table()
        cross_layer_analysis = self.create_feature_cross_layer_analysis()
        
        # Generate README
        self.generate_readme()
        
        # Print final summary
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        print("="*80)
        print("📊 Summary of all layers:")
        print(summary_df.to_string(index=False))
        
        print(f"\n🔍 Cross-layer analysis:")
        print(f"   Consistently stable features: {cross_layer_analysis['consistently_stable_count']}")
        print(f"   Consistently high drift features: {cross_layer_analysis['consistently_high_drift_count']}")
        
        print(f"\n📁 Files generated:")
        print(f"   - COMPREHENSIVE_ANALYSIS_README.md")
        print(f"   - comprehensive_layer_analysis.png")
        print(f"   - comprehensive_layer_analysis.json")
        print(f"   - layer_analysis_summary.csv")
        print(f"   - layer_analysis_summary.html")
        print(f"   - cross_layer_feature_analysis.json")
        
        return self.results, summary_df, cross_layer_analysis

def main():
    """Main function to run the comprehensive analysis"""
    analyzer = ComprehensiveSAEAnalyzer()
    results, summary_df, cross_layer_analysis = analyzer.run_comprehensive_analysis()
    
    print(f"\n✅ Analysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
