#!/usr/bin/env python3
"""
CLI Version - Small Dataset Auto-Interp
=======================================

This script runs Delphi using the CLI approach with a smaller dataset (50 latents)
as requested. It follows the original script structure but with reduced scope.
"""

import subprocess
import sys
import os

def run_delphi_cli():
    """Run Delphi using CLI with smaller dataset."""
    
    # Configuration
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"
    SAE_PATH = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    print("🚀 Running Delphi Auto-Interp CLI (Small Dataset)")
    print("=" * 60)
    print(f"🎯 Base Model: {BASE_MODEL}")
    print(f"🎯 SAE Path: {SAE_PATH}")
    print(f"🎯 Layer: 16")
    print(f"📊 Max Latents: 50 (smaller dataset as requested)")
    
    # Delphi CLI command
    cmd = [
        "python", "-m", "delphi",
        BASE_MODEL,
        SAE_PATH,
        "--n_tokens", "1000000",        # 1M tokens (smaller)
        "--max_latents", "50",          # 50 latents (smaller dataset)
        "--hookpoints", "layers.16",
        "--scorers", "detection",
        "--filter_bos",
        "--name", "llama2_7b_layer16_small_autointerp"
    ]
    
    print(f"\n🔧 Running command: {' '.join(cmd)}")
    
    try:
        # Run Delphi
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Delphi CLI completed successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Delphi CLI failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

if __name__ == "__main__":
    success = run_delphi_cli()
    sys.exit(0 if success else 1)
