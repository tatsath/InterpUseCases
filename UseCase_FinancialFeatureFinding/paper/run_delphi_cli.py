#!/usr/bin/env python3
"""
Delphi CLI Finance Auto-Interp
==============================

This script runs Delphi using the CLI approach with supported flags only.
It demonstrates the default end-to-end pipeline (cache → explain → detection-score).

Based on Delphi's official CLI documentation and supported parameters.
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path

def load_config(config_file="delphi/config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"⚠️  Config file {config_file} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        return {}

def run_delphi_cli(config):
    """Run Delphi using CLI with supported flags only."""
    
    # Default values (from Delphi's official defaults)
    defaults = {
        "model": "meta-llama/Llama-2-7b-hf",
        "sparse_model": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "n_tokens": 5000000,
        "max_latents": 100,
        "hookpoints": ["layers.16"],
        "scorers": ["detection"],
        "filter_bos": True,
        "run_name": "llama2_7b_layer16_finance_autointerp"
    }
    
    # Merge config with defaults
    settings = {**defaults, **config}
    
    print("🚀 Running Delphi Auto-Interp CLI (Finance)")
    print("=" * 60)
    print(f"🎯 Base Model: {settings['model']}")
    print(f"🎯 SAE Path: {settings['sparse_model']}")
    print(f"🎯 Layer: {', '.join(settings['hookpoints'])}")
    print(f"📊 Max Latents: {settings['max_latents']}")
    print(f"🔢 Tokens: {settings['n_tokens']:,}")
    print(f"🎯 Scorers: {', '.join(settings['scorers'])}")
    
    # Build Delphi CLI command with supported flags only
    cmd = [
        "python", "-m", "delphi",
        settings["model"],
        settings["sparse_model"],
        "--n_tokens", str(settings["n_tokens"]),
        "--max_latents", str(settings["max_latents"]),
        "--hookpoints", *settings["hookpoints"],
        "--scorers", *settings["scorers"],
        "--name", settings["run_name"]
    ]
    
    # Add optional flags
    if settings.get("filter_bos", False):
        cmd.append("--filter_bos")
    
    print(f"\n🔧 Running command: {' '.join(cmd)}")
    
    try:
        # Run Delphi
        print("\n🚀 Starting Delphi execution...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Delphi CLI completed successfully!")
        print("\n📊 Output:")
        print(result.stdout)
        
        # Check for results
        run_dir = f"runs/{settings['run_name']}"
        if os.path.exists(run_dir):
            print(f"\n📁 Results saved in: {run_dir}")
            print("📋 Generated files:")
            for file in os.listdir(run_dir):
                file_path = os.path.join(run_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   - {file} ({size:,} bytes)")
        else:
            print(f"\n⚠️  Expected run directory not found: {run_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Delphi CLI failed with exit code {e.returncode}")
        print("\n📤 STDOUT:")
        print(e.stdout)
        print("\n📤 STDERR:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Delphi module not found. Please ensure Delphi is installed.")
        return False

def main():
    """Main execution function."""
    print("🚀 Delphi Finance Auto-Interp CLI Runner")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    if not config:
        print("⚠️  Using default configuration")
    
    # Run Delphi CLI
    success = run_delphi_cli(config)
    
    if success:
        print("\n🎉 CLI execution completed successfully!")
        print("\n📋 Next steps:")
        print("1. Review results in the runs/ directory")
        print("2. For FAISS hard-negatives, use the programmatic approach")
        print("3. For finance-specific prompts, modify the explainer configuration")
    else:
        print("\n❌ CLI execution failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
