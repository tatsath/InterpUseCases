#!/usr/bin/env python3
"""
Delphi Results Analysis Script
==============================

This script provides comprehensive analysis of Delphi Auto-Interp results:
1. F1 score calculation and analysis
2. Explanation quality assessment
3. Latent performance metrics
4. Visualization and reporting
5. Export capabilities

Separate from the main pipeline for better modularity and debugging.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add delphi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'delphi'))

try:
    from delphi.log.result_analysis import (
        compute_confusion, 
        compute_classification_metrics,
        frequency_weighted_f1
    )
    DELPHI_AVAILABLE = True
    print("✅ Delphi's built-in analysis tools available")
except ImportError as e:
    DELPHI_AVAILABLE = False
    print(f"⚠️  Delphi's built-in analysis tools not available: {e}")
    print("Using custom F1 calculation methods")

class DelphiResultsAnalyzer:
    """Comprehensive analysis of Delphi Auto-Interp results."""
    
    def __init__(self, results_dir: str = "runs/llama2_7b_layer16_finance_autointerp"):
        """Initialize analyzer with results directory."""
        self.results_dir = Path(results_dir)
        self.explanations_dir = self.results_dir / "explanations"
        self.scores_dir = self.results_dir / "scores"
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.results = {}
        self.f1_metrics = {}
        self.summary_stats = {}
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Explanations directory: {self.explanations_dir}")
        print(f"📁 Scores directory: {self.scores_dir}")
        print(f"📁 Analysis directory: {self.analysis_dir}")
    
    def load_all_results(self):
        """Load all explanation and scoring results."""
        print("\n🔍 Loading all results...")
        
        # Load explanations
        explanation_files = list(self.explanations_dir.glob("*.json"))
        print(f"📄 Found {len(explanation_files)} explanation files")
        
        for file in explanation_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                latent_id = data.get('latent_id', file.stem)
                self.results[latent_id] = {
                    'explanation': data.get('explanation', 'No explanation'),
                    'timestamp': data.get('timestamp', 'Unknown'),
                    'f1_score': data.get('f1_score'),
                    'f1_metrics': data.get('f1_metrics')
                }
                
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
        
        # Load scores
        score_files = list(self.scores_dir.glob("*.json"))
        print(f"📊 Found {len(score_files)} score files")
        
        for file in score_files:
            try:
                with open(file, 'r') as f:
                    scores = json.load(f)
                
                latent_id = f"layers.16_{file.stem}"
                
                # Calculate F1 metrics from scores
                f1_metrics = self.calculate_f1_from_scores(scores)
                
                if latent_id in self.results:
                    self.results[latent_id]['f1_metrics'] = f1_metrics
                    self.results[latent_id]['f1_score'] = f1_metrics.get('f1_score', 0.0)
                    self.results[latent_id]['scores'] = scores
                
                self.f1_metrics[latent_id] = f1_metrics
                
            except Exception as e:
                print(f"❌ Error loading scores {file}: {e}")
        
        print(f"✅ Loaded {len(self.results)} complete results")
        return self.results
    
    def calculate_f1_from_scores(self, scores: List[Dict]) -> Dict[str, float]:
        """Calculate F1 metrics from raw score data."""
        try:
            # Extract predictions and ground truth
            predictions = []
            ground_truth = []
            
            for score in scores:
                pred = score.get('prediction')
                truth = score.get('activating')
                
                if pred is not None and truth is not None:
                    predictions.append(bool(pred))
                    ground_truth.append(bool(truth))
            
            if not predictions:
                return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
            
            # Calculate confusion matrix
            tp = sum(1 for p, t in zip(predictions, ground_truth) if p and t)
            tn = sum(1 for p, t in zip(predictions, ground_truth) if not p and not t)
            fp = sum(1 for p, t in zip(predictions, ground_truth) if p and not t)
            fn = sum(1 for p, t in zip(predictions, ground_truth) if not p and t)
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(predictions) if predictions else 0.0
            
            return {
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total": len(predictions)
            }
            
        except Exception as e:
            print(f"❌ Error calculating F1 metrics: {e}")
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    
    def analyze_f1_performance(self, f1_cutoff: float = 0.5):
        """Analyze F1 performance across all latents."""
        print(f"\n📊 Analyzing F1 Performance (cutoff: {f1_cutoff})")
        print("=" * 60)
        
        if not self.f1_metrics:
            print("❌ No F1 metrics available. Run load_all_results() first.")
            return
        
        # Calculate summary statistics
        f1_scores = [metrics.get('f1_score', 0.0) for metrics in self.f1_metrics.values()]
        precisions = [metrics.get('precision', 0.0) for metrics in self.f1_metrics.values()]
        recalls = [metrics.get('recall', 0.0) for metrics in self.f1_metrics.values()]
        accuracies = [metrics.get('accuracy', 0.0) for metrics in self.f1_metrics.values()]
        
        self.summary_stats = {
            'total_latents': len(f1_scores),
            'f1_cutoff': f1_cutoff,
            'above_f1_cutoff': sum(1 for f1 in f1_scores if f1 >= f1_cutoff),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'min_f1': np.min(f1_scores),
            'max_f1': np.max(f1_scores),
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_accuracy': np.mean(accuracies)
        }
        
        # Display results
        print(f"📈 F1 Performance Summary:")
        print(f"   Total latents: {self.summary_stats['total_latents']}")
        print(f"   Above F1 cutoff ({f1_cutoff}): {self.summary_stats['above_f1_cutoff']}")
        print(f"   Mean F1: {self.summary_stats['mean_f1']:.3f} ± {self.summary_stats['std_f1']:.3f}")
        print(f"   F1 range: {self.summary_stats['min_f1']:.3f} - {self.summary_stats['max_f1']:.3f}")
        print(f"   Mean precision: {self.summary_stats['mean_precision']:.3f}")
        print(f"   Mean recall: {self.summary_stats['mean_recall']:.3f}")
        print(f"   Mean accuracy: {self.summary_stats['mean_accuracy']:.3f}")
        
        return self.summary_stats
    
    def show_top_performers(self, top_k: int = 5):
        """Show top performing latents by F1 score."""
        print(f"\n🏆 Top {top_k} Latents by F1 Score:")
        print("=" * 80)
        
        # Sort by F1 score
        sorted_latents = sorted(
            self.f1_metrics.items(),
            key=lambda x: x[1].get('f1_score', 0.0),
            reverse=True
        )
        
        for i, (latent_id, metrics) in enumerate(sorted_latents[:top_k]):
            explanation = self.results.get(latent_id, {}).get('explanation', 'No explanation')
            
            print(f"\n🥇 Rank {i+1}: {latent_id}")
            print(f"   F1 Score: {metrics.get('f1_score', 0.0):.3f}")
            print(f"   Precision: {metrics.get('precision', 0.0):.3f}")
            print(f"   Recall: {metrics.get('recall', 0.0):.3f}")
            print(f"   Accuracy: {metrics.get('accuracy', 0.0):.3f}")
            print(f"   TP: {metrics.get('tp', 0)}, TN: {metrics.get('tn', 0)}")
            print(f"   FP: {metrics.get('fp', 0)}, FN: {metrics.get('fn', 0)}")
            print(f"   Explanation: {explanation}")
            print("-" * 60)
    
    def analyze_explanations(self):
        """Analyze explanation quality and patterns."""
        print(f"\n💡 Explanation Analysis:")
        print("=" * 60)
        
        explanations = [result.get('explanation', '') for result in self.results.values()]
        
        # Basic statistics
        explanation_lengths = [len(exp.split()) for exp in explanations]
        unique_explanations = len(set(explanations))
        
        print(f"📝 Explanation Statistics:")
        print(f"   Total explanations: {len(explanations)}")
        print(f"   Unique explanations: {unique_explanations}")
        print(f"   Mean length: {np.mean(explanation_lengths):.1f} words")
        print(f"   Length range: {np.min(explanation_lengths)} - {np.max(explanation_lengths)} words")
        
        # Show all explanations
        print(f"\n📋 All Explanations:")
        for latent_id, result in self.results.items():
            f1_score = result.get('f1_score', 0.0)
            if f1_score is None:
                f1_score = 0.0
            explanation = result.get('explanation', 'No explanation')
            print(f"   {latent_id}: F1={f1_score:.3f} | {explanation}")
        
        return {
            'total_explanations': len(explanations),
            'unique_explanations': unique_explanations,
            'mean_length': np.mean(explanation_lengths),
            'length_range': (np.min(explanation_lengths), np.max(explanation_lengths))
        }
    
    def create_visualizations(self):
        """Create visualizations of the results."""
        print(f"\n📊 Creating Visualizations...")
        
        if not self.f1_metrics:
            print("❌ No F1 metrics available for visualization")
            return
        
        # Prepare data
        f1_scores = [metrics.get('f1_score', 0.0) for metrics in self.f1_metrics.values()]
        precisions = [metrics.get('precision', 0.0) for metrics in self.f1_metrics.values()]
        recalls = [metrics.get('recall', 0.0) for metrics in self.f1_metrics.values()]
        latent_ids = list(self.f1_metrics.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Delphi Auto-Interp Results Analysis', fontsize=16)
        
        # F1 Score Distribution
        axes[0, 0].hist(f1_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('F1 Score Distribution')
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[0, 0].legend()
        
        # Precision vs Recall
        axes[0, 1].scatter(recalls, precisions, alpha=0.7, s=100)
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        
        # F1 Score by Latent
        axes[1, 0].bar(range(len(f1_scores)), f1_scores, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('F1 Score by Latent')
        axes[1, 0].set_xlabel('Latent Index')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_xticks(range(len(latent_ids)))
        axes[1, 0].set_xticklabels([f"L{i}" for i in range(len(latent_ids))], rotation=45)
        
        # Metrics Comparison
        metrics_data = [f1_scores, precisions, recalls]
        metrics_labels = ['F1 Score', 'Precision', 'Recall']
        axes[1, 1].boxplot(metrics_data, labels=metrics_labels)
        axes[1, 1].set_title('Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.analysis_dir / "results_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {plot_file}")
        
        plt.show()
    
    def export_results(self, format: str = "json"):
        """Export analysis results in various formats."""
        print(f"\n📤 Exporting Results ({format.upper()})...")
        
        # Prepare export data
        export_data = {
            'summary_stats': self.summary_stats,
            'f1_metrics': self.f1_metrics,
            'results': self.results,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if format.lower() == "json":
            export_file = self.analysis_dir / "complete_analysis.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"📄 JSON export saved: {export_file}")
            
        elif format.lower() == "csv":
            # Create DataFrame for CSV export
            df_data = []
            for latent_id, metrics in self.f1_metrics.items():
                result = self.results.get(latent_id, {})
                row = {
                    'latent_id': latent_id,
                    'f1_score': metrics.get('f1_score', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'accuracy': metrics.get('accuracy', 0.0),
                    'tp': metrics.get('tp', 0),
                    'tn': metrics.get('tn', 0),
                    'fp': metrics.get('fp', 0),
                    'fn': metrics.get('fn', 0),
                    'total': metrics.get('total', 0),
                    'explanation': result.get('explanation', 'No explanation'),
                    'timestamp': result.get('timestamp', 'Unknown')
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            csv_file = self.analysis_dir / "results_analysis.csv"
            df.to_csv(csv_file, index=False)
            print(f"📊 CSV export saved: {csv_file}")
            
        else:
            print(f"❌ Unsupported format: {format}")
    
    def run_complete_analysis(self, f1_cutoff: float = 0.5, top_k: int = 5):
        """Run complete analysis pipeline."""
        print("🚀 Starting Complete Delphi Results Analysis")
        print("=" * 80)
        
        # Load all results
        self.load_all_results()
        
        # Analyze F1 performance
        self.analyze_f1_performance(f1_cutoff)
        
        # Show top performers
        self.show_top_performers(top_k)
        
        # Analyze explanations
        self.analyze_explanations()
        
        # Create visualizations
        self.create_visualizations()
        
        # Export results
        self.export_results("json")
        self.export_results("csv")
        
        print(f"\n🎉 Complete analysis finished!")
        print(f"📁 All results saved in: {self.analysis_dir}")

def main():
    """Main execution function."""
    print("🔍 Delphi Results Analyzer")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DelphiResultsAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis(f1_cutoff=0.3, top_k=10)  # Lower cutoff to see more results

if __name__ == "__main__":
    main()
