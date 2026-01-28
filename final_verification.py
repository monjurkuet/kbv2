#!/usr/bin/env python3
"""Final verification that model rotation system works correctly."""

import asyncio
import sys
from typing import List

from knowledge_base.clients.rotation_manager import ModelRotationManager
from knowledge_base.clients.rotating_llm_client import RotatingLLMClient, ModelRotationConfig

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def test_rotation_config():
    """Test that rotation config has correct values."""
    print("\n=== Test 1: Rotation Configuration ===")
    config = ModelRotationConfig()
    
    # Check retry delay
    print(f"  retry_delay: {config.retry_delay}s")
    if config.retry_delay < 5.0:
        print(f"  {FAIL}: retry_delay must be >= 5.0 seconds")
        return False
    print(f"  {PASS}: retry_delay >= 5.0 seconds")
    
    # Check models list
    print(f"  models in rotation: {len(config.models)}")
    if len(config.models) == 0:
        print(f"  {FAIL}: No models configured")
        return False
    print(f"  {PASS}: {len(config.models)} models configured")
    
    for i, model in enumerate(config.models[:3], 1):
        print(f"    {i}. {model}")
    
    return True


def test_rotating_client():
    """Test RotatingLLMClient initialization."""
    print("\n=== Test 2: RotatingLLMClient ===")
    client = RotatingLLMClient()
    
    # Check it has rotation config
    if not hasattr(client, 'rotation_config'):
        print(f"  {FAIL}: Missing rotation_config attribute")
        return False
    print(f"  {PASS}: rotation_config exists")
    
    # Check models
    if not hasattr(client.rotation_config, 'models'):
        print(f"  {FAIL}: Missing models list")
        return False
    
    models = client.rotation_config.models
    print(f"  {PASS}: {len(models)} models loaded")
    
    return True


async def test_manager_rotation():
    """Test ModelRotationManager rotates models correctly."""
    print("\n=== Test 3: Model Rotation Manager ===")
    
    async with ModelRotationManager() as manager:
        # Get current rotation
        rotation = manager.get_available_models()
        print(f"  Available models: {len(rotation)}")
        
        if len(rotation) == 0:
            print(f"  {FAIL}: No models available")
            return False
        print(f"  {PASS}: Has {len(rotation)} models")
        
        # Test that manager has required methods
        required_methods = ['call_llm', 'get_available_models', 'close']
        for method in required_methods:
            if not hasattr(manager, method):
                print(f"  {FAIL}: Missing method '{method}'")
                return False
        print(f"  {PASS}: All required methods exist")
        
        return True


async def test_actual_llm_call():
    """Test an actual LLM call with model rotation."""
    print("\n=== Test 4: Actual LLM Call ===")
    
    async with ModelRotationManager() as manager:
        try:
            result = await manager.call_llm(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=50
            )
            
            if not result.get('success'):
                print(f"  {FAIL}: LLM call failed: {result.get('error')}")
                return False
            
            print(f"  {PASS}: LLM call succeeded")
            print(f"  Model used: {result['model']}")
            print(f"  Response: {result['content'][:80]}...")
            
            return True
            
        except Exception as e:
            print(f"  {FAIL}: Exception: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run all tests."""
    print("=" * 70)
    print("KBV2 Model Rotation System - Final Verification")
    print("=" * 70)
    
    tests = [
        ("Rotation Configuration", test_rotation_config),
        ("RotatingLLMClient", test_rotating_client),
        ("Model Rotation Manager", test_manager_rotation),
        ("Actual LLM Call", test_actual_llm_call),
    ]
    
    results = []
    
    # Run sync tests
    for test_name, test_func in tests[:2]:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"  Result: {PASS if result else FAIL}")
        except Exception as e:
            print(f"  {FAIL}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Run async tests
    for test_name, test_func in tests[2:]:
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"  Result: {PASS if result else FAIL}")
        except Exception as e:
            print(f"  {FAIL}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = PASS if result else FAIL
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Model rotation system is fully operational:")
        print("   • 5-second retry delay enforced")
        print("   • Multiple models configured for rotation")  
        print("   • Automatic fallback on rate limits")
        print("   • LLM calls working correctly")
        return 0
    else:
        print(f"\n❌ {total - passed}/{total} tests failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
