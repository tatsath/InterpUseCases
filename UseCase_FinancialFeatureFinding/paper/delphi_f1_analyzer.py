#!/usr/bin/env python3
"""
Delphi Official F1 Score Calculator
===================================

This script implements Delphi's exact F1 calculation methodology:
1. Uses Delphi's official compute_confusion() and compute_classification_metrics()
2. Follows the same data format and processing pipeline
3. Implements frequency-weighted F1 scoring
4. Provides comprehensive metrics matching Delphi's output

Based on delphi/delphi/log/result_analysis.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

# Add delphi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'delphi'))

try:
    from delphi.log.result_analysis import (
        compute_confusion, 
        compute_classification_metrics,
        frequency_weighted_f1,
        load_data,
        get_agg_metrics
    )
    DELPHI_AVAILABLE = True
    print("✅ Delphi's official analysis tools available")
except ImportError as e:
    DELPHI_AVAILABLE = False
    print(f"⚠️  Delphi's official analysis tools not available: {e}")

class DelphiOfficialF1Analyzer:
    """Implements Delphi's exact F1 calculation methodology."""
    
    def __init__(self, results_dir: str = "runs/llama2_7b_layer16_finance_autointerp"):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.scores_dir = self.results_dir / "scores"
        self.analysis_dir = self.results_dir / "delphi_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        print(f"📁 Results directory: {self.results_dir}")
        print(f"📁 Scores directory: {self.scores_dir}")
        print(f"📁 Analysis directory: {self.analysis_dir}")
    
    def convert_scores_to_delphi_format(self) -> pd.DataFrame:
        """Convert our score files to Delphi's expected DataFrame format."""
        print("\n🔄 Converting scores to Delphi format...")
        
        all_data = []
        
        # Load all score files
        score_files = list(self.scores_dir.glob("*.json"))
        print(f"📊 Found {len(score_files)} score files")
        
        for file in score_files:
            try:
                with open(file, 'r') as f:
                    scores = json.load(f)
                
                # Extract latent index from filename
                latent_idx = int(file.stem.split("latent")[-1])
                
                # Convert each score to Delphi format
                for score in scores:
                    # Delphi expects these specific column names
                    delphi_row = {
                        "text": "".join(score.get("str_tokens", [])),
                        "distance": score.get("distance", -1.0),
                        "activating": bool(score.get("activating", False)),
                        "prediction": bool(score.get("prediction", False)) if score.get("prediction") is not None else None,
                        "probability": score.get("probability"),
                        "correct": score.get("correct"),
                        "activations": score.get("activations", []),
                        "latent_idx": latent_idx,
                        "module": "layers.16",  # Our module name
                        "score_type": "detection"  # Default score type
                    }
                    all_data.append(delphi_row)
                
                print(f"✅ Converted {file.name}: {len(scores)} examples")
                
            except Exception as e:
                print(f"❌ Error converting {file}: {e}")
        
        # Create DataFrame in Delphi format
        df = pd.DataFrame(all_data)
        print(f"📊 Created DataFrame with {len(df)} rows")
        
        return df
    
    def compute_delphi_f1_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute F1 metrics using Delphi's official methodology."""
        print("\n📊 Computing F1 metrics using Delphi's official methodology...")
        
        if df.empty:
            print("❌ No data available for F1 computation")
            return {}
        
        # Group by latent for per-latent analysis
        latent_results = {}
        
        for latent_idx, group_df in df.groupby("latent_idx"):
            print(f"\n🔍 Analyzing latent {latent_idx}...")
            
            # Use Delphi's official compute_confusion
            try:
                confusion = compute_confusion(group_df)
                print(f"   Confusion matrix: TP={confusion['true_positives']}, TN={confusion['true_negatives']}, FP={confusion['false_positives']}, FN={confusion['false_negatives']}")
                
                # Use Delphi's official compute_classification_metrics
                metrics = compute_classification_metrics(confusion)
                print(f"   F1 Score: {metrics['f1_score']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall: {metrics['recall']:.3f}")
                print(f"   Balanced Accuracy: {metrics['accuracy']:.3f}")
                
                latent_results[f"latent_{latent_idx}"] = {
                    "confusion": confusion,
                    "metrics": metrics,
                    "total_examples": len(group_df)
                }
                
            except Exception as e:
                print(f"   ❌ Error computing metrics for latent {latent_idx}: {e}")
                latent_results[f"latent_{latent_idx}"] = {
                    "error": str(e),
                    "total_examples": len(group_df)
                }
        
        return latent_results
    
    def compute_aggregate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute aggregate metrics using Delphi's get_agg_metrics."""
        print("\n📊 Computing aggregate metrics...")
        
        try:
            # Use Delphi's official get_agg_metrics
            agg_metrics = get_agg_metrics(df, counts=None)
            
            if not agg_metrics.empty:
                # Get the first (and likely only) score type
                score_type_summary = agg_metrics.iloc[0]
                
                aggregate_results = {
                    "score_type": score_type_summary["score_type"],
                    "f1_score": score_type_summary["f1_score"],
                    "precision": score_type_summary["precision"],
                    "recall": score_type_summary["recall"],
                    "accuracy": score_type_summary["accuracy"],
                    "true_positives": score_type_summary["true_positives"],
                    "true_negatives": score_type_summary["true_negatives"],
                    "false_positives": score_type_summary["false_positives"],
                    "false_negatives": score_type_summary["false_negatives"],
                    "total_examples": score_type_summary["total_examples"],
                    "total_positives": score_type_summary["total_positives"],
                    "total_negatives": score_type_summary["total_negatives"],
                    "true_positive_rate": score_type_summary["true_positive_rate"],
                    "true_negative_rate": score_type_summary["true_negative_rate"],
                    "false_positive_rate": score_type_summary["false_positive_rate"],
                    "false_negative_rate": score_type_summary["false_negative_rate"]
                }
                
                print(f"✅ Aggregate F1 Score: {aggregate_results['f1_score']:.3f}")
                print(f"✅ Aggregate Precision: {aggregate_results['precision']:.3f}")
                print(f"✅ Aggregate Recall: {aggregate_results['recall']:.3f}")
                print(f"✅ Aggregate Accuracy: {aggregate_results['accuracy']:.3f}")
                
                return aggregate_results
            else:
                print("❌ No aggregate metrics computed")
                return {}
                
        except Exception as e:
            print(f"❌ Error computing aggregate metrics: {e}")
            return {}
    
    def display_delphi_results(self, latent_results: Dict[str, Any], aggregate_results: Dict[str, Any]):
        """Display results in Delphi's official format."""
        print(f"\n🎯 Delphi Official F1 Analysis Results")
        print("=" * 80)
        
        # Display per-latent results
        print(f"\n📊 Per-Latent Results:")
        print("-" * 60)
        
        sorted_latents = sorted(
            latent_results.items(),
            key=lambda x: x[1].get("metrics", {}).get("f1_score", 0.0) if "metrics" in x[1] else 0.0,
            reverse=True
        )
        
        for latent_id, result in sorted_latents:
            if "metrics" in result:
                metrics = result["metrics"]
                confusion = result["confusion"]
                
                print(f"\n🏆 {latent_id.upper()}:")
                print(f"   F1 Score: {metrics['f1_score']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall: {metrics['recall']:.3f}")
                print(f"   Balanced Accuracy: {metrics['accuracy']:.3f}")
                print(f"   TP: {confusion['true_positives']}, TN: {confusion['true_negatives']}")
                print(f"   FP: {confusion['false_positives']}, FN: {confusion['false_negatives']}")
                print(f"   Total Examples: {result['total_examples']}")
            else:
                print(f"\n❌ {latent_id.upper()}: {result.get('error', 'Unknown error')}")
        
        # Display aggregate results
        if aggregate_results:
            print(f"\n📈 Aggregate Results:")
            print("-" * 60)
            print(f"Overall F1 Score: {aggregate_results['f1_score']:.3f}")
            print(f"Overall Precision: {aggregate_results['precision']:.3f}")
            print(f"Overall Recall: {aggregate_results['recall']:.3f}")
            print(f"Overall Accuracy: {aggregate_results['accuracy']:.3f}")
            print(f"True Positive Rate: {aggregate_results['true_positive_rate']:.3f}")
            print(f"True Negative Rate: {aggregate_results['true_negative_rate']:.3f}")
            print(f"False Positive Rate: {aggregate_results['false_positive_rate']:.3f}")
            print(f"False Negative Rate: {aggregate_results['false_negative_rate']:.3f}")
            print(f"Total Examples: {aggregate_results['total_examples']}")
            print(f"Total Positives: {aggregate_results['total_positives']}")
            print(f"Total Negatives: {aggregate_results['total_negatives']}")
    
    def export_delphi_results(self, latent_results: Dict[str, Any], aggregate_results: Dict[str, Any]):
        """Export results in Delphi's format."""
        print(f"\n📤 Exporting Delphi Official Results...")
        
        export_data = {
            "delphi_analysis": {
                "methodology": "Official Delphi F1 calculation using compute_confusion and compute_classification_metrics",
                "timestamp": pd.Timestamp.now().isoformat(),
                "per_latent_results": latent_results,
                "aggregate_results": aggregate_results
            }
        }
        
        # Export JSON
        json_file = self.analysis_dir / "delphi_official_f1_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"📄 Delphi results exported: {json_file}")
        
        # Export CSV summary
        csv_data = []
        for latent_id, result in latent_results.items():
            if "metrics" in result:
                metrics = result["metrics"]
                confusion = result["confusion"]
                row = {
                    "latent_id": latent_id,
                    "f1_score": metrics["f1_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "accuracy": metrics["accuracy"],
                    "true_positives": confusion["true_positives"],
                    "true_negatives": confusion["true_negatives"],
                    "false_positives": confusion["false_positives"],
                    "false_negatives": confusion["false_negatives"],
                    "total_examples": result["total_examples"]
                }
                csv_data.append(row)
        
        if csv_data:
            df_summary = pd.DataFrame(csv_data)
            csv_file = self.analysis_dir / "delphi_official_f1_summary.csv"
            df_summary.to_csv(csv_file, index=False)
            print(f"📊 Delphi summary exported: {csv_file}")
    
    def run_delphi_analysis(self):
        """Run complete Delphi official F1 analysis."""
        print("🚀 Starting Delphi Official F1 Analysis")
        print("=" * 80)
        
        if not DELPHI_AVAILABLE:
            print("❌ Delphi's official analysis tools not available")
            return False
        
        # Convert scores to Delphi format
        df = self.convert_scores_to_delphi_format()
        
        if df.empty:
            print("❌ No data available for analysis")
            return False
        
        # Compute per-latent F1 metrics
        latent_results = self.compute_delphi_f1_metrics(df)
        
        # Compute aggregate metrics
        aggregate_results = self.compute_aggregate_metrics(df)
        
        # Display results
        self.display_delphi_results(latent_results, aggregate_results)
        
        # Export results
        self.export_delphi_results(latent_results, aggregate_results)
        
        print(f"\n🎉 Delphi Official F1 Analysis Complete!")
        print(f"📁 Results saved in: {self.analysis_dir}")
        
        return True

def main():
    """Main execution function."""
    print("🔍 Delphi Official F1 Score Calculator")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = DelphiOfficialF1Analyzer()
    
    # Run Delphi official analysis
    success = analyzer.run_delphi_analysis()
    
    if success:
        print("\n✅ Delphi official F1 analysis completed successfully!")
        print("📊 This uses the exact same methodology as Delphi's official pipeline")
    else:
        print("\n❌ Delphi official F1 analysis failed")

if __name__ == "__main__":
    main()
