#!/usr/bin/env python3
"""End-to-end test simulating real user workflow"""

import sys
sys.path.insert(0, 'src')
import asyncio
import json
import time

async def test_e2e_user_workflow():
    """Simulate complete user workflow end-to-end"""
    
    print("="*80)
    print("🎭 E2E TEST: Real User Simulation")
    print("="*80)
    print("User actions:")
    print("1. Opens browser to http://localhost:5173")
    print("2. Frontend connects to backend WebSocket")
    print("3. User clicks 'Add Document'")
    print("4. User selects file")
    print("5. User clicks 'Upload'")
    print("6. User watches progress updates")
    print("7. User sees completion message")
    print("="*80)
    
    # Step 1: Backend health check (UI does this on load)
    print("\n[Step 1/6] Checking backend health...")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get("http://localhost:8765/health")
        assert resp.status_code == 200, f"Backend unhealthy: {resp.status_code}"
        print(f"✅ Backend healthy: {resp.json()}")
    
    # Step 2: WebSocket connection (UI connects on load)
    print("\n[Step 2/6] Connecting WebSocket...")
    import websockets
    uri = "ws://127.0.0.1:8765/ws"
    async with websockets.connect(uri) as ws:
        print("✅ WebSocket connected")
        
        # Step 3: User creates document
        print("\n[Step 3/6] Creating test document...")
        test_content = """
        SpaceX successfully launched Falcon Heavy with Tesla Roadster payload.
        Elon Musk celebrated the achievement on social media.
        Launch took place from Kennedy Space Center on February 6, 2018.
        """
        with open("./testdata/falcon_heavy.txt", "w") as f:
            f.write(test_content.strip())
        print("✅ Document created")
        
        # Step 4: User clicks "Upload"
        print("\n[Step 4/6] User clicks 'Upload'...")
        start_time = time.time()
        request = {
            "method": "kbv2/ingest_document",
            "params": {"file_path": "./testdata/falcon_heavy.txt", "document_name": "Falcon Heavy Launch"},
            "id": f"user-test-{int(start_time)}"
        }
        print(f"  Sending: {json.dumps(request)}")
        await ws.send(json.dumps(request))
        print("✅ Upload request sent")
        
        # Step 5: Monitor progress (UI displays updates)
        print("\n[Step 5/6] Monitoring progress...")
        print("-"*80)
        print(f"{'Time':>6} | {'Stage':>5} | {'Status':>10} | Message")
        print("-"*80)
        
        updates = []
        while time.time() - start_time < 90:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2)
                data = json.loads(msg)
                updates.append(data)
                
                if data.get('type') == 'progress':
                    elapsed = time.time() - start_time
                    stage = data.get('stage', '?')
                    status = data.get('status', '?')
                    message = data.get('message', '')
                    print(f"{elapsed:6.2f}s | {str(stage):>5} | {status:>10} | {message[:40]}")
                elif 'error' in data:
                    print(f"❌ ERROR: {data['error']}")
                    return False
                elif 'result' in data:
                    print(f"✅ COMPLETE!")
                    break
            except asyncio.TimeoutError:
                continue
        
        # Step 6: Verify results
        print("\n" + "="*80)
        print("[Step 6/6] Verifying results...")
        completed = [u for u in updates if u.get('type') == 'progress' and u.get('status') == 'completed']
        print(f"✅ Completed stages: {len(completed)}")
        print(f"✅ Total updates: {len(updates)}")
        print(f"✅ Total time: {time.time() - start_time:.1f}s")
        
        return len(completed) >= 8

# Run test
try:
    result = asyncio.run(test_e2e_user_workflow())
    print("\n" + "="*80)
    if result:
        print("🎉 END-TO-END TEST: ✅ PASSED")
        print("   All stages completed successfully")
        print("   Progress tracking works correctly")
    else:
        print("❌ END-TO-END TEST: FAILED")
    print("="*80)
    sys.exit(0 if result else 1)
except Exception as e:
    print(f"\n❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
