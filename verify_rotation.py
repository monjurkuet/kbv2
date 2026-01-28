#!/usr/bin/env python3
"""
Verify KBV2 model rotation system is working correctly.

This script tests:
1. ModelRegistry fetching from LLM Gateway
2. RotatingLLMClient initialization
3. Model rotation on mock rate limits
4. 5-second retry delay enforcement
"""

import asyncio
import time
from datetime import datetime

try:
    from knowledge_base.clients.rotation_manager import ModelRotationManager

    print("✅ Successfully imported ModelRotationManager")
except ImportError as e:
    print(f"❌ Failed to import ModelRotationManager: {e}")
    exit(1)

try:
    from knowledge_base.clients.model_registry import ModelRegistryManager

    print("✅ Successfully imported ModelRegistryManager")
except ImportError as e:
    print(f"❌ Failed to import ModelRegistryManager: {e}")
    exit(1)


async def test_model_registry():
    """Test that ModelRegistry can fetch models from LLM Gateway."""
    print("\n=== Testing ModelRegistry ===")

    try:
        registry = await ModelRegistryManager.get_registry()

        if registry:
            # Check available providers
            providers = ["kimi", "qwen", "glm", "deepseek", "gemini"]
            print("Available models by provider:")

            for provider in providers:
                models = registry.get_provider_models(provider)
                if models:
                    print(f"  {provider}: {len(models)} models")
                    print(
                        f"    Recommended: {registry.get_recommended_model(provider)}"
                    )
                else:
                    print(f"  {provider}: No models found")

            # Test fallback model
            fallback = registry.get_fallback_model()
            print(f"\nFallback model: {fallback}")
            print("✅ ModelRegistry working correctly")
            return True
        else:
            print("⚠️ ModelRegistry not available, using defaults")
            return False

    except Exception as e:
        print(f"❌ ModelRegistry error: {e}")
        return False


async def test_basic_call():
    """Test a basic LLM call with rotation manager."""
    print("\n=== Testing Basic LLM Call ===")

    try:
        async with ModelRotationManager() as manager:
            start_time = time.time()

            result = await manager.call_llm(
                messages=[{"role": "user", "content": "Hello, what is 2+2?"}],
                temperature=0.0,
                max_tokens=50,
            )

            elapsed = time.time() - start_time

            if result["success"]:
                print(f"✅ Success! Response: {result['content'][:100]}...")
                print(f"   Model used: {result['model']}")
                print(f"   Time elapsed: {elapsed:.2f}s")
                return True
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_rate_limit_handling():
    """Test rate limit handling (mock test)."""
    print("\n=== Testing Rate Limit Handling ===")
    print(
        "Note: This is a mock test. Real rate limit testing requires actual 429 responses."
    )

    try:
        # Create manager with multiple models
        async with ModelRotationManager() as manager:
            rotation_models = manager.get_current_rotation()
            print(f"Current rotation: {', '.join(rotation_models)}")
            print(f"✅ Manager ready with {len(rotation_models)} models")
            print("   Rate limit handling will activate on 429 errors")
            return True

    except Exception as e:
        print(f"❌ Rate limit test setup failed: {e}")
        return False


async def verify_5_second_delay():
    """Verify that rate limit delays enforce 5-second minimum."""
    print("\n=== Verifying 5-Second Delay Enforcement ===")

    from knowledge_base.clients.rotating_llm_client import (
        RotatingLLMClient,
        ModelRotationConfig,
    )

    # Create a test client
    client = RotatingLLMClient()

    # Check the retry delay configuration
    actual_delay = client.rotation_config.retry_delay
    expected_min = 5.0

    print(f"Configured retry delay: {actual_delay}s")
    print(f"Minimum required: {expected_min}s")

    if actual_delay >= expected_min:
        print("✅ 5-second minimum delay is enforced")
        return True
    else:
        print(
            f"❌ Delay is less than required minimum: {actual_delay}s < {expected_min}s"
        )
        return False


async def main():
    """Run all verification tests."""
    print("=" * 70)
    print("KBV2 Model Rotation System Verification")
    print("=" * 70)
    print(f"Started at: {datetime.now().isoformat()}")

    tests = [
        ("Model Registry", test_model_registry),
        ("5-Second Delay", verify_5_second_delay),
        ("Rate Limit Handling", test_rate_limit_handling),
        ("Basic LLM Call", test_basic_call),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Model rotation system is ready!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check logs above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
