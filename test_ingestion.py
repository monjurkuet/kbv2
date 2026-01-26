#!/usr/bin/env python3
import asyncio
import websockets
import json
import sys

async def test_real_ingestion():
    uri = "ws://localhost:8765/ws"
    file_path = "./testdata/2019-07-03_Consolidation_Cheat_Sheet.md"
    
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("✓ WebSocket connected")
        
        # Send ingest request
        request = {
            "method": "kbv2/ingest_document",
            "params": {
                "file_path": file_path,
                "document_name": "Test Document"
            },
            "id": "test-123"
        }
        
        print(f"\nSending ingestion request...")
        print(f"File: {file_path}")
        
        await websocket.send(json.dumps(request))
        
        # Receive response
        response = await websocket.recv()
        print(f"\nResponse received: {response[:200]}...")
        
        # Keep listening for progress updates
        print("\nListening for progress updates...")
        for i in range(20):  # Listen for up to 20 messages/30 seconds
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.5)
                data = json.loads(message)
                
                if isinstance(data, dict) and data.get("type") == "progress":
                    stage = data.get("stage", "?")
                    status = data.get("status", "?")
                    msg = data.get("message", "")
                    duration = data.get("duration", 0)
                    print(f"  [{i}] Stage {stage} - {status}: {msg} ({duration:.2f}s)")
                else:
                    print(f"  [{i}] {data}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"  [{i}] Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(test_real_ingestion())
