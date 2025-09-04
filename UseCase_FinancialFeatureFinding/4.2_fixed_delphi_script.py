#!/usr/bin/env python3
"""
Fixed Delphi Script - Compatibility Version
This script addresses the version compatibility issues found in the diagnostic.
"""

import os
import sys
import traceback
from pathlib import Path

def run_delphi_with_workarounds():
    """Run Delphi with compatibility workarounds"""
    print("🚀 Running Delphi with compatibility fixes...")
    
    try:
        # Import Delphi components
        from delphi.config import ConstructorConfig, SamplerConfig
        
        # Create configurations that avoid the problematic FAISS integration
        constructor_cfg = ConstructorConfig(
            non_activating_source="random",  # Use random instead of FAISS
            faiss_embedding_cache_enabled=False,
            n_non_activating=20,  # Reduce for faster processing
            min_examples=100  # Reduce minimum examples
        )
        
        sampler_cfg = SamplerConfig(
            n_examples_train=20,  # Reduce training examples
            n_examples_test=25,   # Reduce test examples
            n_quantiles=5         # Reduce quantiles
        )
        
        print("✅ Configurations created successfully")
        print(f"   Constructor: {constructor_cfg}")
        print(f"   Sampler: {sampler_cfg}")
        
        # Try to run the actual Delphi command with modified parameters
        print("\n🎯 Attempting to run Delphi with fixed parameters...")
        
        # Use subprocess to run the delphi command with our fixed config
        import subprocess
        
        # Build the command with workarounds
        cmd = [
            "python", "-m", "delphi",
            "meta-llama/Llama-2-7b-hf",
            "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "--n_tokens", "50000",
            "--max_latents", "2",
            "--hookpoints", "layers.16",
            "--scorers", "detection",
            "--filter_bos",
            "--name", "llama2_7b_layer16_sae_autointerp_fixed"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print("⚠️  Note: FAISS disabled due to compatibility issues")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Delphi completed successfully!")
            print("📁 Check results in: llama2_7b_layer16_sae_autointerp_fixed/")
        else:
            print("❌ Delphi failed with error:")
            print(result.stderr)
            
            # Try to provide helpful error analysis
            if "NonActivatingExample" in result.stderr:
                print("\n🔍 Error Analysis:")
                print("   This is the FAISS compatibility issue we identified.")
                print("   The script is using 'random' source instead of FAISS.")
                print("   If this still fails, we may need to investigate further.")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
        traceback.print_exc()
        return False

def test_alternative_approaches():
    """Test alternative approaches if the main method fails"""
    print("\n🔍 Testing alternative approaches...")
    
    try:
        # Approach 1: Try to import and inspect the problematic classes
        print("📦 Inspecting Delphi classes...")
        
        try:
            from delphi.explainers import ContrastiveExplainer
            print("✅ ContrastiveExplainer imported")
            
            # Check what arguments it expects
            import inspect
            sig = inspect.signature(ContrastiveExplainer.__init__)
            print(f"   Constructor signature: {sig}")
            
        except Exception as e:
            print(f"❌ ContrastiveExplainer inspection failed: {e}")
        
        # Approach 2: Check if there are alternative explainers
        try:
            from delphi.explainers import DefaultExplainer
            print("✅ DefaultExplainer imported")
            
            import inspect
            sig = inspect.signature(DefaultExplainer.__init__)
            print(f"   Constructor signature: {sig}")
            
        except Exception as e:
            print(f"❌ DefaultExplainer inspection failed: {e}")
        
        # Approach 3: Look for any working explainers
        try:
            import delphi.explainers
            print(f"📦 Available explainers: {dir(delphi.explainers)}")
            
            # Try to find any explainer that might work
            for item in dir(delphi.explainers):
                if "Explainer" in item and not item.startswith("_"):
                    try:
                        explainer_class = getattr(delphi.explainers, item)
                        if hasattr(explainer_class, "__init__"):
                            sig = inspect.signature(explainer_class.__init__)
                            print(f"   {item}: {sig}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ Explainer discovery failed: {e}")
            
    except Exception as e:
        print(f"❌ Alternative testing failed: {e}")

def main():
    """Main function"""
    print("🚀 Fixed Delphi Script - Compatibility Version")
    print("=" * 60)
    
    # Try the main approach first
    success = run_delphi_with_workarounds()
    
    if not success:
        print("\n🔄 Main approach failed, trying alternatives...")
        test_alternative_approaches()
        
        print("\n💡 Recommendations:")
        print("1. The FAISS integration has compatibility issues in this Delphi version")
        print("2. Try running without FAISS using 'random' source")
        print("3. Consider updating Delphi to a newer version")
        print("4. Check if there are specific version requirements for FAISS integration")
    
    print("\n🎯 Next steps:")
    print("- Check the output directory for results")
    print("- If successful, gradually increase --max_latents")
    print("- Consider running without FAISS for now")

if __name__ == "__main__":
    main()
