#!/usr/bin/env python3
"""
FAISS Debug Script for Delphi
This script tests different configurations to identify and fix FAISS integration issues.
"""

import os
import sys
import traceback
from pathlib import Path

def test_delphi_imports():
    """Test if Delphi and its dependencies can be imported"""
    print("🔍 Testing Delphi imports...")
    
    try:
        import delphi
        print("✅ Delphi imported successfully")
        
        # Check available modules
        print(f"📦 Available Delphi modules: {dir(delphi)}")
        
        if hasattr(delphi, 'explainers'):
            print("✅ Explainers module available")
            print(f"   Available explainers: {dir(delphi.explainers)}")
        
        if hasattr(delphi, 'config'):
            print("✅ Config module available")
            print(f"   Available configs: {dir(delphi.config)}")
            
    except ImportError as e:
        print(f"❌ Failed to import Delphi: {e}")
        return False
    
    return True

def test_faiss_imports():
    """Test FAISS-related imports"""
    print("\n🔍 Testing FAISS imports...")
    
    try:
        import faiss
        print(f"✅ FAISS imported successfully (version: {faiss.__version__})")
        
        import sentence_transformers
        print(f"✅ Sentence transformers imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import FAISS dependencies: {e}")
        return False

def test_delphi_config():
    """Test Delphi configuration options"""
    print("\n🔍 Testing Delphi configuration...")
    
    try:
        from delphi.config import ConstructorConfig
        
        # Test different FAISS configurations
        configs_to_test = [
            ("FAISS enabled", {"non_activating_source": "FAISS"}),
            ("FAISS with embedding model", {
                "non_activating_source": "FAISS",
                "faiss_embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }),
            ("Random source", {"non_activating_source": "random"}),
            ("None source", {"non_activating_source": None}),
        ]
        
        for name, config_dict in configs_to_test:
            try:
                config = ConstructorConfig(**config_dict)
                print(f"✅ {name}: Configuration created successfully")
                print(f"   Config: {config}")
            except Exception as e:
                print(f"❌ {name}: Configuration failed - {e}")
                
    except Exception as e:
        print(f"❌ Failed to test configurations: {e}")
        traceback.print_exc()

def test_explainer_creation():
    """Test creating different explainers"""
    print("\n🔍 Testing explainer creation...")
    
    try:
        from delphi.explainers import DefaultExplainer, ContrastiveExplainer
        
        # Test DefaultExplainer
        try:
            explainer = DefaultExplainer()
            print("✅ DefaultExplainer created successfully")
        except Exception as e:
            print(f"❌ DefaultExplainer failed: {e}")
        
        # Test ContrastiveExplainer
        try:
            explainer = ContrastiveExplainer()
            print("✅ ContrastiveExplainer created successfully")
        except Exception as e:
            print(f"❌ ContrastiveExplainer failed: {e}")
            
    except Exception as e:
        print(f"❌ Failed to test explainers: {e}")
        traceback.print_exc()

def test_minimal_delphi_run():
    """Test a minimal Delphi run without FAISS"""
    print("\n🔍 Testing minimal Delphi run...")
    
    try:
        # Import necessary components
        from delphi.latents import LatentCache
        from delphi.config import ConstructorConfig, SamplerConfig
        
        print("✅ Basic imports successful")
        
        # Test configuration creation
        constructor_cfg = ConstructorConfig(
            non_activating_source="random",  # Use random instead of FAISS
            faiss_embedding_cache_enabled=False
        )
        
        sampler_cfg = SamplerConfig()
        
        print("✅ Configurations created successfully")
        print(f"   Constructor config: {constructor_cfg}")
        print(f"   Sampler config: {sampler_cfg}")
        
    except Exception as e:
        print(f"❌ Minimal run failed: {e}")
        traceback.print_exc()

def test_faiss_workaround():
    """Test potential FAISS workarounds"""
    print("\n🔍 Testing FAISS workarounds...")
    
    try:
        from delphi.config import ConstructorConfig
        
        # Test 1: Disable FAISS completely
        try:
            config1 = ConstructorConfig(
                non_activating_source=None,
                faiss_embedding_cache_enabled=False
            )
            print("✅ Workaround 1 (FAISS disabled): Success")
        except Exception as e:
            print(f"❌ Workaround 1 failed: {e}")
        
        # Test 2: Use random source with FAISS disabled
        try:
            config2 = ConstructorConfig(
                non_activating_source="random",
                faiss_embedding_cache_enabled=False
            )
            print("✅ Workaround 2 (Random source): Success")
        except Exception as e:
            print(f"❌ Workaround 2 failed: {e}")
        
        # Test 3: Minimal configuration
        try:
            config3 = ConstructorConfig()
            print("✅ Workaround 3 (Default config): Success")
            print(f"   Default non_activating_source: {config3.non_activating_source}")
        except Exception as e:
            print(f"❌ Workaround 3 failed: {e}")
            
    except Exception as e:
        print(f"❌ Workaround testing failed: {e}")
        traceback.print_exc()

def check_delphi_version():
    """Check Delphi version and compatibility"""
    print("\n🔍 Checking Delphi version...")
    
    try:
        import delphi
        if hasattr(delphi, '__version__'):
            print(f"📦 Delphi version: {delphi.__version__}")
        else:
            print("📦 Delphi version: Unknown")
        
        # Check if it's a git installation
        delphi_path = Path(delphi.__file__).parent
        if (delphi_path / '.git').exists():
            print(f"📦 Delphi installed from git: {delphi_path}")
            
            # Try to get git info
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], 
                    cwd=delphi_path, 
                    capture_output=True, 
                    text=True
                )
                if result.returncode == 0:
                    print(f"📦 Git commit: {result.stdout.strip()[:8]}")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Version check failed: {e}")

def main():
    """Main diagnostic function"""
    print("🚀 Delphi FAISS Debug Script")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_delphi_imports,
        test_faiss_imports,
        test_delphi_config,
        test_explainer_creation,
        test_minimal_delphi_run,
        test_faiss_workaround,
        check_delphi_version
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
        
        print("-" * 50)
    
    print("\n🎯 Summary of findings:")
    print("1. Check if FAISS is properly installed")
    print("2. Verify Delphi version compatibility")
    print("3. Try using 'random' source instead of FAISS")
    print("4. Check if there are any missing dependencies")

if __name__ == "__main__":
    main()
