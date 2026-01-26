#!/usr/bin/env python3
"""End-to-end frontend test with Playwright - simulates real user interaction."""

import time
import sys
from playwright.sync_api import sync_playwright

def test_document_ingestion():
    """Test full document ingestion through the UI like a real user."""
    
    print("="*70)
    print("🎭 PLAYWRIGHT E2E TEST: Real User Simulation")
    print("="*70)
    print("Steps:")
    print("1. Launch browser")
    print("2. Navigate to frontend")
    print("3. Click 'Add Document'")
    print("4. Wait for progress updates")
    print("5. Verify all stages complete")
    print("6. Check document appears in UI")
    print("="*70)
    
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(
            headless=False,  # Show the browser so we can see what's happening
            args=['--start-maximized']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        
        page = context.new_page()
        
        # Step 1: Navigate to frontend
        print("\n📍 Navigating to http://localhost:5173...")
        try:
            page.goto("http://localhost:5173", wait_until='networkidle', timeout=30000)
            print("✅ Frontend loaded")
        except Exception as e:
            print(f"❌ Failed to load frontend: {e}")
            browser.close()
            return False
        
        # Take screenshot for debugging
        page.screenshot(path="/tmp/frontend_loaded.png")
        
        # Step 2: Wait for connection status
        print("\n🔌 Checking WebSocket connection status...")
        try:
            # Wait for connection status indicator
            page.wait_for_selector("[data-testid='connection-status']", timeout=10000)
            status = page.locator("[data-testid='connection-status']").inner_text()
            print(f"Connection status: {status}")
            
            if "CONNECTED" in status:
                print("✅ Backend connected")
            else:
                print(f"⚠️  Connection status: {status}")
        except:
            print("⚠️  Could not find connection status (might be okay)")
        
        # Step 3: Click 
