#!/usr/bin/env python3
"""
Enhanced Finance Auto-Interp Pipeline
=====================================

This implements the full-scale approach described in the research paper:
- 50 samples per feature for interpretation
- 200 samples per feature for scoring (non-overlapping)
- Top decile + balanced negative sampling
- Detection F1 calculation
- Full feature coverage (all 400 latents)
"""

import os
import json
import numpy as np
import subprocess
import yaml
from pathlib import Path

class EnhancedFinancePipeline:
    """Enhanced pipeline matching the research paper approach."""
    
    def __init__(self, config_file="delphi/config.yaml"):
        self.config = self.load_config(config_file)
        self.setup_paths()
        
    def load_config(self, config_file):
        """Load configuration."""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                # Ensure all required keys exist by merging with defaults
                default_config = self.get_default_config()
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Get enhanced configuration matching research paper."""
        return {
            "model": "meta-llama/Llama-2-7b-hf",
            "sparse_model": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "hookpoints": ["layers.16"],
            "max_latents": 400,  # All features
            "n_tokens": 500000,  # Increased for better sampling
            "interpretation_samples": 50,  # Per feature for interpretation
            "scoring_samples": 200,        # Per feature for scoring
            "top_decile_ratio": 0.1,      # Top 10% for positives
            "balance_pos_neg": True,       # Equal positive/negative
            "explainer": {
                "model": "Qwen/Qwen2.5-72B-Instruct",
                "provider": "offline",
                "max_len": 8192,
                "temperature": 0.2
            },
            "scorers": ["detection"],
            "output_dir": "runs",
            "run_name": "enhanced_finance_autointerp_full_scale"
        }
    
    def setup_paths(self):
        """Setup output paths."""
        self.output_dir = Path(self.config.get("output_dir", "runs"))
        self.run_dir = self.output_dir / self.config.get("run_name", "enhanced_finance_autointerp_full_scale")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.run_dir}")
    
    def run_enhanced_delphi(self):
        """Run Delphi with enhanced parameters matching research paper."""
        print("🚀 Running Enhanced Delphi Pipeline (Research Paper Approach)")
        print("=" * 70)
        
        # Enhanced Delphi command matching research paper
        cmd = [
            "python", "-m", "delphi",
            self.config["model"],
            self.config["sparse_model"],
            "--n_tokens", str(self.config["n_tokens"]),
            "--max_latents", str(self.config["max_latents"]),
            "--hookpoints", *self.config["hookpoints"],
            "--scorers", *self.config["scorers"],
            "--name", self.config["run_name"],
            "--filter_bos",
            # Enhanced sampling parameters
            "--n_examples_train", str(self.config["interpretation_samples"]),
            "--n_examples_test", str(self.config["scoring_samples"]),
            "--train_type", "top",  # Top decile sampling
            "--test_type", "quantiles",
            "--ratio_top", str(self.config["top_decile_ratio"]),
            "--min_examples", str(self.config["interpretation_samples"]),
            "--n_non_activating", str(self.config["interpretation_samples"])  # Balanced negatives
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print(f"📊 Target: {self.config['max_latents']} features")
        print(f"📝 Interpretation samples: {self.config['interpretation_samples']} per feature")
        print(f"🎯 Scoring samples: {self.config['scoring_samples']} per feature")
        print(f"⚖️  Balanced sampling: {self.config['balance_pos_neg']}")
        
        try:
            print("\n🚀 Starting enhanced execution...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("✅ Enhanced pipeline completed successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    
    def calculate_detection_f1(self):
        """Calculate Detection F1 scores from generated results."""
        print("\n🔍 Calculating Detection F1 Scores...")
        
        # Find detection score files
        detection_dir = self.run_dir / "scores" / "detection"
        if not detection_dir.exists():
            print("❌ Detection scores directory not found")
            return False
        
        f1_scores = {}
        
        # Process each latent's detection scores
        for score_file in detection_dir.glob("*.txt"):
            latent_name = score_file.stem
            print(f"📊 Processing {latent_name}...")
            
            try:
                # Parse detection scores and calculate F1
                f1_score = self.parse_detection_scores(score_file)
                f1_scores[latent_name] = f1_score
                
            except Exception as e:
                print(f"⚠️  Error processing {latent_name}: {e}")
                f1_scores[latent_name] = None
        
        # Save F1 scores
        f1_output = self.run_dir / "detection_f1_scores.json"
        with open(f1_output, 'w') as f:
            json.dump(f1_scores, f, indent=2)
        
        print(f"✅ F1 scores saved to: {f1_output}")
        
        # Calculate statistics
        valid_scores = [s for s in f1_scores.values() if s is not None]
        if valid_scores:
            print(f"📊 F1 Score Statistics:")
            print(f"   - Features processed: {len(valid_scores)}")
            print(f"   - Mean F1: {np.mean(valid_scores):.3f}")
            print(f"   - Median F1: {np.median(valid_scores):.3f}")
            print(f"   - Std F1: {np.std(valid_scores):.3f}")
            print(f"   - Features with F1 > 0.65: {sum(1 for s in valid_scores if s > 0.65)}")
        
        return True
    
    def parse_detection_scores(self, score_file):
        """Parse detection scores and calculate F1."""
        with open(score_file, 'r') as f:
            content = f.read()
        
        # Parse JSON lines
        scores = []
        for line in content.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    scores.append({
                        'activating': data.get('activating', False),
                        'prediction': data.get('prediction', False),
                        'correct': data.get('correct', False)
                    })
                except json.JSONDecodeError:
                    continue
        
        if not scores:
            return None
        
        # Calculate F1 score
        tp = sum(1 for s in scores if s['activating'] and s['prediction'])
        fp = sum(1 for s in scores if not s['activating'] and s['prediction'])
        fn = sum(1 for s in scores if s['activating'] and not s['prediction'])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def run_full_pipeline(self):
        """Run the complete enhanced pipeline."""
        print("🚀 Enhanced Finance Auto-Interp Pipeline")
        print("=" * 70)
        print("🎯 Target: Full-scale feature interpretation (400 latents)")
        print("📊 Sampling: 50 samples per feature for interpretation")
        print("🎯 Scoring: 200 samples per feature for detection F1")
        print("⚖️  Strategy: Top decile + balanced negative sampling")
        
        # Run enhanced Delphi
        if not self.run_enhanced_delphi():
            return False
        
        # Calculate F1 scores
        if not self.calculate_detection_f1():
            return False
        
        print("\n🎉 Enhanced pipeline completed successfully!")
        print(f"📁 Results: {self.run_dir}")
        print(f"📊 F1 scores: {self.run_dir}/detection_f1_scores.json")
        
        return True

def main():
    """Main execution."""
    pipeline = EnhancedFinancePipeline()
    success = pipeline.run_full_pipeline()
    
    if not success:
        print("\n❌ Enhanced pipeline failed")
        exit(1)
    
    print("\n🎯 Next steps:")
    print("1. Review F1 scores for all 400 features")
    print("2. Analyze high-quality interpretations (F1 > 0.65)")
    print("3. Use results for research paper analysis")

if __name__ == "__main__":
    main()
