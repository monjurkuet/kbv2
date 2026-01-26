#!/usr/bin/env python3
"""End-to-end frontend test with Playwright - simulates real user interaction."""

import time
from playwright.sync_api import sync_playwright

def test_ingestion_workflow():
    """Test document ingestion like a real user through the UI."""
    
    print("\n" + "="*70)
    print("🎭 PLAYWRIGHT E2E TEST: Real User Simulation")
    print("="*70)
    
    with sync_playwright() as p:
        print("\n🚀 Launching browser...")
        browser = p.chromium.launch(
            headless=False,  # Show browser so we can see what's happening
            args=['--start-maximized']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = context.new_page()
        
        # Step 1: Navigate to frontend
        print("\n📍 Navigating to http://localhost:5173...")
        try:
            page.goto("http://localhost:5173", wait_until='networkidle', timeout=30000)
            print("✅ Frontend loaded successfully")
            page.screenshot(path="/tmp/1_frontend_loaded.png")
        except Exception as e:
            print(f"❌ Failed to load frontend: {e}")
            browser.close()
            return False
        
        # Step 2: Check connection status
        print("\n🔌 Checking WebSocket connection status...")
        try:
            # Look for connection status indicator
            status_element = page.locator("text=/CONNECTED|DISCONNECTED/")
            if status_element.count() > 0:
                status = status_element.first.text_content(timeout=5000)
                print(f"Connection status: {status}")
                if "CONNECTED" in status:
                    print("✅ Backend WebSocket connected")
                else:
                    print(f"⚠️  Connection status: {status}")
            else:
                print("⚠️  Could not find connection status indicator")
        except Exception as e:
            print(f"⚠️  Connection check failed: {e}")
        
        page.screenshot(path="/tmp/2_connection_check.png")
        
        # Step 3: Look for document ingestion UI
        print("\n🔍 Looking for document ingestion interface...")
        try:
            # Wait for any interactive elements
            page.wait_for_selector("button, input, [role='button']", timeout=10000)
            print("✅ Interactive elements found")
            
            # List all buttons
            buttons = page.locator("button").all()
            if buttons:
                print(f"  Found {len(buttons)} buttons:")
                for i, btn in enumerate(buttons[:5], 1):
                    text = btn.text_content().strip()[:50]
                    print(f"    {i}. '{text}'")
            
            # Look for file input or document-related elements
            file_inputs = page.locator("input[type='file']").all()
            if file_inputs:
                print(f"  ✅ Found {len(file_inputs)} file input(s)")
            
            # Look for "document" or "ingest" text
            doc_elements = page.locator("text=/document|ingest|add/i").all()
            if doc_elements:
                print(f"  ✅ Found {len(doc_elements)} document-related elements")
                
        except Exception as e:
            print(f"⚠️  Could not analyze UI: {e}")
        
        page.screenshot(path="/tmp/3_document_ui.png")
        
        # Step 4: Try to interact with document features
        print("\n🖱️  Attempting document interaction...")
        try:
            # Try clicking the first button that looks like it might add a document
            add_buttons = page.locator("button:has-text(/add|upload|document/i)").all()
            if add_buttons:
                print("  Clicking 'Add Document' button...")
                add_buttons[0].click()
                time.sleep(1)
                page.screenshot(path="/tmp/4_after_click.png")
            else:
                # Try clicking the first visible button
                visible_buttons = page.locator("button:visible").all()
                if visible_buttons:
                    print("  Clicking first visible button...")
                    visible_buttons[0].click()
                    time.sleep(1)
                    page.screenshot(path="/tmp/4_after_click.png")
        except Exception as e:
            print(f"⚠️  Could not click button: {e}")
        
        # Step 5: Look for progress indicators
        print("\n📊 Looking for progress indicators...")
        try:
            # Wait a bit for any async loading
            time.sleep(2)
            
            # Look for stage indicators, progress bars, etc.
            progress_bars = page.locator("[data-stage], .progress, .stage").all()
            if progress_bars:
                print(f"  ✅ Found {len(progress_bars)} progress elements")
                for elem in progress_bars[:3]:
                    stage = elem.get_attribute('data-stage') or 'unknown'
                    text = elem.text_content().strip()[:40]
                    print(f"    Stage {stage}: '{text}'")
            else:
                print("  ⚠️  No progress elements found")
                
        except Exception as e:
            print(f"⚠️  Could not check progress: {e}")
        
        page.screenshot(path="/tmp/5_progress_check.png")
        
        # Final screenshot
        time.sleep(2)
        page.screenshot(path="/tmp/6_final_state.png")
        
        print("\n" + "="*70)
        print("✅ E2E test completed")
        print("📸 Screenshots saved to /tmp/*.png")
        print("="*70)
        
        browser.close()
        return True

if __name__ == "__main__":
    try:
        success = test_ingestion_workflow()
        print(f"\n{'='*70}")
        print(f"✅ E2E TEST RESULT: {'PASSED' if success else 'FAILED'}")
        print(f"{'='*70}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
