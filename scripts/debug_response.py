#!/usr/bin/env python3
"""Debug the actual response structure from LLM calls."""

import asyncio
from knowledge_base.clients.rotation_manager import ModelRotationManager

async def main():
    print("Debug: Testing LLM call and printing full response structure\n")
    
    async with ModelRotationManager() as manager:
        try:
            result = await manager.call_llm(
                messages=[{"role": "user", "content": "What is 2+2?"}],
                max_tokens=50
            )
            
            print("Result keys:", list(result.keys()))
            print("\nFull result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
                
            # If there's an error, show it
            if not result.get('success'):
                print(f"\n❌ Error: {result.get('error')}")
                if 'models_tried' in result:
                    print(f"Models tried: {result['models_tried']}")
            else:
                print(f"\n✅ Success with model: {result['model']}")
                print(f"Response: {result['content'][:100]}...")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
