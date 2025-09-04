#!/usr/bin/env python3
"""
Test script to verify Pipeline and Pipe functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'delphi'))

async def test_pipeline():
    """Test basic pipeline functionality."""
    try:
        print("🔧 Testing Delphi imports...")
        
        # Test basic imports
        from delphi.pipeline import Pipeline, Pipe, process_wrapper
        print("✅ Pipeline imports successful")
        
        # Test Pipe creation
        def test_function(x):
            return x * 2
        
        pipe = Pipe(test_function)
        print("✅ Pipe creation successful")
        
        # Test Pipeline creation
        def dummy_loader():
            return [1, 2, 3]
        
        pipeline = Pipeline(dummy_loader, pipe)
        print("✅ Pipeline creation successful")
        
        print("🎉 All pipeline tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_explainer():
    """Test explainer functionality."""
    try:
        print("\n🔧 Testing explainer imports...")
        
        from delphi.explainers import ContrastiveExplainer
        print("✅ ContrastiveExplainer import successful")
        
        # Test that ContrastiveExplainer doesn't accept system_prompt
        try:
            # This should fail
            explainer = ContrastiveExplainer(None, system_prompt="test")
            print("❌ ContrastiveExplainer incorrectly accepted system_prompt")
            return False
        except TypeError:
            print("✅ ContrastiveExplainer correctly rejected system_prompt")
        
        print("🎉 All explainer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Explainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Delphi Pipeline and Components")
    print("=" * 50)
    
    pipeline_ok = await test_pipeline()
    explainer_ok = await test_explainer()
    
    if pipeline_ok and explainer_ok:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Need to fix issues before running main script.")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
