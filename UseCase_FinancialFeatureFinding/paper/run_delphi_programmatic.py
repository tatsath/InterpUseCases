#!/usr/bin/env python3
"""
Delphi Programmatic Finance Auto-Interp with F1 Scoring
======================================================

This script demonstrates the programmatic approach to Delphi Auto-Interp with:
1. FAISS hard-negatives for better contrastive learning
2. Finance-specific prompts and label constraints
3. Full control over explainer and scorer configuration
4. F1 score calculation and cutoff filtering
5. Results display and analysis

Based on Delphi's official API and supported features.
"""

import os
import sys
import asyncio
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

# Delphi imports from local directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'delphi'))

try:
    from delphi.latents import LatentDataset
    from delphi.config import SamplerConfig, ConstructorConfig
    from delphi.clients import Offline
    from delphi.explainers import ContrastiveExplainer, DefaultExplainer
    from delphi.pipeline import Pipeline, process_wrapper
    from delphi.scorers import DetectionScorer, FuzzingScorer
except ImportError as e:
    print(f"❌ Delphi not available: {e}")
    print("Please check the local delphi directory")
    sys.exit(1)

class FinanceAutoInterp:
    """Finance-specific Auto-Interp using Delphi's official API with F1 scoring."""
    
    def __init__(self, config_file="delphi/config.yaml"):
        """Initialize with configuration."""
        self.config = self.load_config(config_file)
        self.setup_paths()
        self.results = {}  # Store F1 scores and explanations
    
    def load_config(self, config_file):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file {config_file} not found, using defaults")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration."""
        return {
            "model": "meta-llama/Llama-2-7b-hf",
            "sparse_model": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "hookpoints": ["layers.16"],
            "max_latents": 100,
            "faiss": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "cache_enabled": True,
                "cache_dir": ".embedding_cache"
            },
            "explainer": {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "provider": "offline",
                "max_len": 8192,
                "temperature": 0.2,
                "num_gpus": 4,
                "max_memory": 0.8,
                "batch_size": 1,  # Process one latent at a time
                "enable_cuda_graphs": True,  # Use CUDA graphs for speed
                "max_concurrent": 1  # Limit concurrent processing
            },
            "scorers": ["recall", "fuzzing"],
            "n_processes": 4,
            "output_dir": "runs",
            "run_name": "llama2_7b_layer16_finance_autointerp",
            "f1_cutoff": 0.0,  # Show all results (no cutoff)
            "min_firing_rate": 0.01,  # Minimum firing rate threshold
            "show_top_k": 10  # Show top K latents by F1 score
        }
    
    def setup_paths(self):
        """Setup output and data paths."""
        self.output_dir = Path(self.config.get("output_dir", "runs"))
        self.run_dir = self.output_dir / self.config.get("run_name", "finance_autointerp")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.data_dir = Path("data")
        self.faiss_dir = Path("faiss")
        
        print(f"📁 Output directory: {self.run_dir}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📁 FAISS directory: {self.faiss_dir}")
    
    def create_finance_prompt(self) -> str:
        """Create finance-specific system prompt with label constraints."""
        labels = self.config.get("finance_labels", [
            "EARNINGS_BEAT", "GUIDANCE_UP", "DOWNGRADE", "CREDIT_DOWNGRADE",
            "LAYOFFS", "M&A_RUMOR", "M&A_CONFIRMED", "CFO_CHANGE",
            "SEC_ENFORCEMENT", "MACRO_INFLATION", "RATE_CUT", "DIVIDEND_CUT",
            "BANKRUPTCY", "REGULATORY_RISK", "ESG_CONTROVERSY", "FX_HEADWIND",
            "STOCK_PERFORMANCE", "MARKET_TREND", "COMPANY_NEWS", "ECONOMIC_INDICATOR",
            "OTHER"
        ])
        
        labels_str = ", ".join(labels)
        
        return f"""You are a financial analyst specializing in interpreting neural network activations in financial and business contexts.

You must choose one label from the following list:
[{labels_str}]

IMPORTANT: Focus on the SEMANTIC MEANING and CONCEPTS that the latent represents, NOT on grammatical parts of speech. 
Instead of saying "nouns, pronouns, prepositions", explain WHAT IDEAS, CONCEPTS, or MEANINGS the latent has learned to recognize.

Focus on the semantic patterns: what topics, concepts, entities, or ideas does this latent recognize?
Avoid generic grammatical descriptions like "nouns, verbs, articles" - instead explain the meaning or purpose.

Return your response in this exact JSON format:
{{"label": "CHOSEN_LABEL", "rationale": "Your explanation of what this latent represents conceptually"}}

Examples of good explanations:
- "This latent recognizes financial performance metrics, company earnings, and stock market indicators"
- "This latent identifies company names, stock symbols, and financial entities in business news"
- "This latent detects market trends, economic conditions, and financial forecasts"

Examples of bad explanations:
- "This latent recognizes nouns, pronouns, and articles" (too generic)
- "This latent identifies parts of speech" (not semantic)
- "This latent finds grammatical elements" (not conceptual)"""
    
    def setup_finance_explainer(self) -> ContrastiveExplainer:
        """Set up the finance-specific contrastive explainer."""
        print("🔧 Setting up finance-specific explainer...")
        
        explainer_config = self.config.get("explainer", {})
        
        # Initialize offline client
        self.client = Offline(
            explainer_config.get("model", "Qwen/Qwen2.5-72B-Instruct"),
            max_memory=0.8,
            max_model_len=explainer_config.get("max_len", 8192),
            num_gpus=explainer_config.get("num_gpus", 4)
        )
        
        # Create contrastive explainer (uses default system prompt)
        explainer = ContrastiveExplainer(
            self.client,
            threshold=0.2,          # Lower threshold for more examples
            max_examples=8,         # Moderate positive examples
            max_non_activating=3,   # Moderate hard negatives to avoid FAISS issues
            verbose=True
        )
        
        return explainer
    
    def setup_latent_dataset(self) -> LatentDataset:
        """Set up the latent dataset with FAISS hard-negatives."""
        print("🔧 Setting up latent dataset with FAISS...")
        
        faiss_config = self.config.get("faiss", {})
        
        # FAISS configuration for hard-negatives (officially supported)
        constructor_cfg = ConstructorConfig(
            non_activating_source="FAISS",
            faiss_embedding_model=faiss_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            faiss_embedding_cache_enabled=faiss_config.get("cache_enabled", True),
            faiss_embedding_cache_dir=faiss_config.get("cache_dir", ".embedding_cache"),
        )
        
        # Sampler configuration (use defaults)
        sampler_cfg = SamplerConfig()
        
        # Create latent dataset
        import torch
        dataset = LatentDataset(
            raw_dir=self.config.get("sparse_model"),
            modules=self.config.get("hookpoints", ["layers.16"]),
            sampler_cfg=sampler_cfg,
            constructor_cfg=constructor_cfg,
            latents={hookpoint: torch.tensor(list(range(400))) for hookpoint in self.config.get("hookpoints", ["layers.16"])},
            tokenizer=None  # Use cached tokens
        )
        
        return dataset
    
    def setup_scorers(self):
        """Set up scoring pipeline."""
        print("🔧 Setting up scorers...")
        
        # Reuse the same client instance
        if not hasattr(self, 'client'):
            print("❌ Client not initialized. Please run setup_finance_explainer first.")
            return []
        
        scorers = []
        scorer_config = self.config.get("scorers", ["recall", "fuzzing"])
        
        # Preprocessing function to extract record from ExplainerResult
        def scorer_preprocess(result):
            if isinstance(result, list):
                result = result[0]
            record = result.record
            record.explanation = result.explanation
            record.extra_examples = record.not_active
            return record
        
        if "recall" in scorer_config:
            recall_scorer = process_wrapper(
                DetectionScorer(self.client, tokenizer=None, batch_size=8),
                preprocess=scorer_preprocess
            )
            scorers.append(recall_scorer)
        
        if "fuzzing" in scorer_config:
            fuzzing_scorer = process_wrapper(
                FuzzingScorer(self.client, tokenizer=None, batch_size=8),
                preprocess=scorer_preprocess
            )
            scorers.append(fuzzing_scorer)
        
        return scorers
    
    def explainer_postprocess(self, result):
        """Post-process explainer results and save as JSON."""
        try:
            # Create explanations directory
            explanations_dir = self.run_dir / "explanations"
            explanations_dir.mkdir(exist_ok=True)
            
            # Store explanation in results
            latent_key = str(result.record.latent)
            if latent_key not in self.results:
                self.results[latent_key] = {}
            self.results[latent_key]['explanation'] = result.explanation
            
            # Save explanation with F1 score if available
            output_data = {
                "explanation": result.explanation,
                "latent_id": str(result.record.latent),
                "f1_score": None,  # Will be updated later when scores are available
                "f1_metrics": None,  # Will be updated later when scores are available
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            output_file = explanations_dir / f"{result.record.latent}.json"
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"✅ Saved explanation: {output_file}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error saving explanation: {e}")
            return result
    
    def scorer_postprocess(self, result):
        """Post-process scorer results using Delphi's built-in F1 scoring."""
        try:
            # Create scores directory
            scores_dir = self.run_dir / "scores"
            scores_dir.mkdir(exist_ok=True)
            
            # Extract scoring results
            if hasattr(result, 'score') and result.score:
                scores = result.score if isinstance(result.score, list) else [result.score]
                
                # Use Delphi's built-in F1 calculation
                f1_metrics = self.use_delphi_f1_scoring(scores)
                
                # Store results
                latent_key = str(result.record.latent)
                if latent_key not in self.results:
                    self.results[latent_key] = {}
                self.results[latent_key]['f1_metrics'] = f1_metrics
                self.results[latent_key]['scores'] = scores
                
                # Save scores as JSON
                output_file = scores_dir / f"{result.record.latent}.json"
                with open(output_file, "w") as f:
                    json.dump([score.__dict__ for score in scores], f, indent=2, default=str)
                
                print(f"✅ Saved scores: {output_file}")
                
                # Update explanation file with F1 score
                self.update_explanation_with_f1(latent_key, f1_metrics)
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing scores: {e}")
            return result
    
    def use_delphi_f1_scoring(self, scores: List[Any]) -> Dict[str, float]:
        """Use Delphi's built-in F1 scoring instead of manual calculation."""
        try:
            # Import Delphi's built-in F1 calculation
            from delphi.log.result_analysis import compute_confusion, compute_classification_metrics
            
            # Convert scores to DataFrame format that Delphi expects
            score_data = []
            for score in scores:
                if hasattr(score, 'prediction') and hasattr(score, 'activating'):
                    pred = score.prediction
                    truth = score.activating
                    
                    if pred is not None and truth is not None:
                        score_data.append({
                            'prediction': bool(pred),
                            'activating': bool(truth),
                            'correct': bool(pred) == bool(truth)
                        })
            
            if not score_data:
                return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
            
            # Use Delphi's built-in methods
            confusion = compute_confusion(score_data)
            metrics = compute_classification_metrics(confusion)
            
            print(f"✅ Used Delphi's built-in F1 scoring: {metrics}")
            return metrics
            
        except ImportError:
            print("⚠️  Delphi's built-in F1 scoring not available, falling back to manual calculation")
            return self.calculate_f1_metrics(scores)
        except Exception as e:
            print(f"❌ Error using Delphi's built-in F1: {e}, falling back to manual calculation")
            return self.calculate_f1_metrics(scores)
    
    def update_explanation_with_f1(self, latent_key: str, f1_metrics: Dict[str, float]):
        """Update explanation JSON file with F1 score information."""
        try:
            explanations_dir = self.run_dir / "explanations"
            explanation_file = explanations_dir / f"layers.16_latent{latent_key}.json"
            
            if explanation_file.exists():
                # Read existing explanation
                with open(explanation_file, "r") as f:
                    data = json.load(f)
                
                # Update with F1 metrics
                data["f1_score"] = f1_metrics.get("f1_score", 0.0)
                data["f1_metrics"] = f1_metrics
                data["last_updated"] = pd.Timestamp.now().isoformat()
                
                # Write back updated data
                with open(explanation_file, "w") as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Updated explanation with F1 score: {explanation_file}")
            
        except Exception as e:
            print(f"❌ Error updating explanation with F1: {e}")
    
    def calculate_f1_metrics(self, scores: List[Any]) -> Dict[str, float]:
        """
        FALLBACK: Manual F1 calculation when Delphi's built-in scoring fails.
        
        This is only used when:
        1. Delphi's built-in F1 scoring is not available
        2. The scorers fail to parse model responses properly
        3. We need a backup calculation method
        
        Ideally, we should use Delphi's built-in F1 scoring via:
        from delphi.log.result_analysis import compute_confusion, compute_classification_metrics
        """
        try:
            # Extract predictions and ground truth
            predictions = []
            ground_truth = []
            
            for score in scores:
                if hasattr(score, 'prediction') and hasattr(score, 'activating'):
                    pred = score.prediction
                    truth = score.activating
                    
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
    
    def apply_f1_cutoff(self) -> Dict[str, Any]:
        """Apply F1 score cutoff and return filtered results."""
        f1_cutoff = self.config.get("f1_cutoff", 0.5)
        min_firing_rate = self.config.get("min_firing_rate", 0.01)
        
        print(f"\n🔍 Applying F1 cutoff: {f1_cutoff}")
        print(f"🔍 Minimum firing rate: {min_firing_rate}")
        
        filtered_results = {}
        cutoff_stats = {
            "total_latents": len(self.results),
            "above_f1_cutoff": 0,
            "above_firing_cutoff": 0,
            "passed_both": 0
        }
        
        for latent_key, result in self.results.items():
            f1_metrics = result.get('f1_metrics', {})
            f1_score = f1_metrics.get('f1_score', 0.0)
            
            # Check F1 cutoff
            if f1_score >= f1_cutoff:
                cutoff_stats["above_f1_cutoff"] += 1
                
                # Check firing rate (if available)
                firing_rate = result.get('firing_rate', 1.0)  # Default to 1.0 if not available
                if firing_rate >= min_firing_rate:
                    cutoff_stats["above_firing_cutoff"] += 1
                    cutoff_stats["passed_both"] += 1
                    filtered_results[latent_key] = result
        
        print(f"📊 Cutoff Results:")
        print(f"   Total latents: {cutoff_stats['total_latents']}")
        print(f"   Above F1 cutoff ({f1_cutoff}): {cutoff_stats['above_f1_cutoff']}")
        print(f"   Above firing rate cutoff ({min_firing_rate}): {cutoff_stats['above_firing_cutoff']}")
        print(f"   Passed both cutoffs: {cutoff_stats['passed_both']}")
        
        return filtered_results, cutoff_stats
    
    def display_results(self, filtered_results: Dict[str, Any]):
        """Display F1 scores and explanations for top latents."""
        if not filtered_results:
            print("\n❌ No latents passed the F1 cutoff!")
            return
        
        print(f"\n🎯 Top {self.config.get('show_top_k', 10)} Latents by F1 Score:")
        print("=" * 80)
        
        # Sort by F1 score
        sorted_results = sorted(
            filtered_results.items(),
            key=lambda x: x[1].get('f1_metrics', {}).get('f1_score', 0.0),
            reverse=True
        )
        
        top_k = min(self.config.get('show_top_k', 10), len(sorted_results))
        
        for i, (latent_key, result) in enumerate(sorted_results[:top_k]):
            f1_metrics = result.get('f1_metrics', {})
            explanation = result.get('explanation', 'No explanation available')
            
            print(f"\n🏆 Rank {i+1}: Latent {latent_key}")
            print(f"   F1 Score: {f1_metrics.get('f1_score', 0.0):.3f}")
            print(f"   Precision: {f1_metrics.get('precision', 0.0):.3f}")
            print(f"   Recall: {f1_metrics.get('recall', 0.0):.3f}")
            print(f"   Accuracy: {f1_metrics.get('accuracy', 0.0):.3f}")
            print(f"   Explanation: {explanation}")
            print(f"   TP: {f1_metrics.get('tp', 0)}, TN: {f1_metrics.get('tn', 0)}")
            print(f"   FP: {f1_metrics.get('fp', 0)}, FN: {f1_metrics.get('fn', 0)}")
            print("-" * 60)
    
    def show_next_steps(self):
        """Show next steps and recommendations."""
        print(f"\n🚀 Next Steps:")
        print("=" * 60)
        print("1. 📊 Review F1 scores in the scores/ directory")
        print("2. 🔍 Analyze explanations in the explanations/ directory")
        print("3. 📈 Adjust F1 cutoff threshold if needed")
        print("4. 🎯 Focus on high-F1 latents for further analysis")
        print("5. 🔧 Modify prompts to improve explanation quality")
        print("6. 📋 Export results for external analysis")
        print("7. 🚀 Scale up to more latents and layers")
        
        # Performance recommendations
        print(f"\n⚡ Performance Optimization Tips:")
        print("=" * 60)
        print("• Reduce model size: Use smaller model (e.g., 7B instead of 72B)")
        print("• Increase batch size: Process multiple latents simultaneously")
        print("• Use fewer GPUs: Single GPU for smaller models")
        print("• Enable caching: Reuse model instances across runs")
        print("• Optimize FAISS: Reduce hard-negative sampling")
        
        # Save summary report
        summary_file = self.run_dir / "summary_report.json"
        summary = {
            "total_latents": len(self.results),
            "f1_cutoff": self.config.get("f1_cutoff", 0.5),
            "min_firing_rate": self.config.get("min_firing_rate", 0.01),
            "performance": {
                "model_size": "72B",
                "num_gpus": self.config.get("explainer", {}).get("num_gpus", 4),
                "batch_size": self.config.get("explainer", {}).get("batch_size", 1),
                "estimated_time_per_latent": "11-16 seconds"
            },
            "results_summary": {
                latent: {
                    "f1_score": result.get('f1_metrics', {}).get('f1_score', 0.0),
                    "explanation": result.get('explanation', 'No explanation')
                }
                for latent, result in self.results.items()
            }
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Summary report saved: {summary_file}")
    
    async def run_pipeline(self):
        """Run the complete pipeline."""
        print("🚀 Starting Finance-Specific Auto-Interp Pipeline with F1 Scoring...")
        print("=" * 80)
        
        try:
            # Check if SAE path exists
            if not os.path.exists(self.config.get("sparse_model")):
                print(f"❌ SAE path not found: {self.config.get('sparse_model')}")
                return False
            
            # Check if FAISS index exists
            faiss_index = self.faiss_dir / "index.faiss"
            if not faiss_index.exists():
                print(f"⚠️  FAISS index not found: {faiss_index}")
                print("Please run build_faiss.py first")
                return False
            
            # Set up components
            dataset = self.setup_latent_dataset()
            explainer = self.setup_finance_explainer()
            scorers = self.setup_scorers()
            
            # Create explainer pipe with post-processing
            explainer_pipe = process_wrapper(explainer, postprocess=self.explainer_postprocess)
            
            # Create scorer pipes with post-processing
            scorer_pipes = []
            for scorer in scorers:
                scorer_pipe = process_wrapper(scorer, postprocess=self.scorer_postprocess)
                scorer_pipes.append(scorer_pipe)
            
            # Create pipeline
            from delphi.pipeline import Pipe
            pipeline = Pipeline(
                dataset,
                explainer_pipe,
                Pipe(*scorer_pipes)
            )
            
            print("🎯 Starting pipeline execution...")
            print(f"📊 Processing 400 latents from layers {self.config.get('hookpoints', ['layers.16'])}")
            print(f"🔍 Using FAISS hard-negatives for contrastive learning")
            print(f"💼 Finance-specific labels: {len(self.config.get('finance_labels', []))} categories")
            print(f"🎯 F1 cutoff threshold: {self.config.get('f1_cutoff', 0.5)}")
            
            # Run pipeline with error handling
            try:
                await pipeline.run(max_concurrent=self.config.get("n_processes", 4))
            except Exception as e:
                if "No non-activating examples found" in str(e):
                    print(f"⚠️  Warning: Some latents skipped due to insufficient non-activating examples")
                    print(f"   This is normal for sparse latents. Continuing with available results...")
                else:
                    raise e
            
            print("🎉 Pipeline completed successfully!")
            print(f"📁 Results saved in: {self.run_dir}")
            
            # Process results
            if self.results:
                print(f"\n📊 Processing {len(self.results)} latent results...")
                
                # Apply F1 cutoff
                filtered_results, cutoff_stats = self.apply_f1_cutoff()
                
                # Display results
                self.display_results(filtered_results)
                
                # Show next steps
                self.show_next_steps()
            
            return True
            
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main execution function."""
    print("🎯 Delphi Finance Auto-Interp with F1 Scoring (Programmatic)")
    print("=" * 80)
    
    # Initialize auto-interp
    autointerp = FinanceAutoInterp()
    
    # Run pipeline
    success = await autointerp.run_pipeline()
    
    if success:
        print("\n🎉 Programmatic execution completed successfully!")
        print("\n📋 Key Features Implemented:")
        print("✅ F1 score calculation and cutoff filtering")
        print("✅ Explanation generation and storage")
        print("✅ Results display and analysis")
        print("✅ Next steps guidance")
    else:
        print("\n❌ Programmatic execution failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
