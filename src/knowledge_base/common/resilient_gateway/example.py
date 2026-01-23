"""Example usage of the Resilient LLM Gateway."""

import asyncio
import logging

from knowledge_base.common.resilient_gateway.gateway import (
    ResilientGatewayClient,
    ResilientGatewayConfig,
)


async def example_usage():
    """Example of how to use the resilient gateway."""

    # Configure the resilient gateway
    config = ResilientGatewayConfig(
        url="http://localhost:8317/v1/",
        api_key="your-api-key-here",  # Use empty string if no API key needed
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
        timeout=120.0,
        # Circuit breaker settings
        circuit_breaker_failure_threshold=5,
        circuit_breaker_recovery_timeout=60,
        circuit_breaker_success_threshold=3,
        # Retry settings
        retry_max_attempts=3,
        retry_base_delay=1.0,
        retry_max_delay=60.0,
        retry_jitter=True,
        retry_on_status_codes=[429, 502, 503, 504],
        # Model switching settings
        model_switching_enabled=True,
        fallback_models=["gpt-3.5-turbo", "gpt-4o-mini", "gemini-2.5-flash"],
        # Metrics settings
        enable_metrics=True,
    )

    # Create the resilient gateway client
    gateway = ResilientGatewayClient(config)

    try:
        # Example 1: Basic chat completion
        print("=== Example 1: Basic Chat Completion ===")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"},
        ]

        response = await gateway.chat_completion(messages)
        print(f"Response: {response.choices[0]['message']['content']}")

        # Example 2: Generate text with system prompt
        print("\n=== Example 2: Generate Text ===")
        text = await gateway.generate_text(
            prompt="Write a short poem about resilience.",
            system_prompt="You are a creative poet focused on themes of resilience and persistence.",
        )
        print(f"Generated text: {text}")

        # Example 3: JSON mode generation
        print("\n=== Example 3: JSON Mode Generation ===")
        json_response = await gateway.generate_text(
            prompt="List 3 programming languages and their main use cases.",
            json_mode=True,
        )
        print(f"JSON response: {json_response}")

        # Example 4: Check metrics
        print("\n=== Example 4: Metrics ===")
        metrics = gateway.get_metrics()
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Successful requests: {metrics['successful_requests']}")
        print(f"Success rate: {metrics['success_rate']}%")
        print(f"Rate limited requests: {metrics['rate_limited_requests']}")
        print(f"Retry attempts: {metrics['retry_attempts']}")
        print(f"Model switches: {metrics['model_switches']}")

        # Example 5: Model-specific metrics
        print("\nModel-specific metrics:")
        for model, model_metrics in metrics["model_metrics"].items():
            print(f"  {model}: {model_metrics}")

    except Exception as e:
        print(f"Error during gateway usage: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Always close the gateway to clean up resources
        await gateway.close()


async def advanced_example():
    """Advanced example with error handling and monitoring."""

    config = ResilientGatewayConfig(
        url="http://localhost:8317/v1/",
        api_key="",
        model="gpt-4o",
        # More aggressive circuit breaking for testing
        circuit_breaker_failure_threshold=2,
        circuit_breaker_recovery_timeout=5,
        retry_max_attempts=2,
        retry_base_delay=0.5,
        model_switching_enabled=True,
    )

    gateway = ResilientGatewayClient(config)

    try:
        # Example with error handling
        messages = [
            {
                "role": "user",
                "content": "What are the key principles of resilient system design?",
            }
        ]

        # This will automatically handle:
        # - Rate limiting (429 errors) by switching models
        # - Server errors (502, 503, 504) with retries
        # - Circuit breaking to prevent cascading failures
        # - Metrics collection for monitoring
        response = await gateway.chat_completion(
            messages=messages, temperature=0.7, max_tokens=500
        )

        print("Advanced example response:")
        print(response.choices[0]["message"]["content"])

        # Print metrics to see how resilience features performed
        metrics = gateway.get_metrics()
        print(f"\nMetrics after advanced example:")
        for key, value in metrics.items():
            if key != "model_metrics":
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error in advanced example: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await gateway.close()


if __name__ == "__main__":
    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)

    print("Running basic example...")
    asyncio.run(example_usage())

    print("\nRunning advanced example...")
    asyncio.run(advanced_example())

    print("\nExamples completed!")
