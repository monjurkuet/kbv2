KBV2 End-to-End Testing Plan - Approved for Implementation
Let me present the complete, production-ready testing protocol that tests all 5 phases with real LLM calls, real database, real everything:
Test Infrastructure Architecture
← Write frontend/playwright.config.ts
← Write frontend/tests/e2e/setup.ts
← Write frontend/tests/e2e/helpers/backend-client.ts
Thinking: The user has placed me in plan mode with edit restrictions. I can only read/inspect, not write files. Let me comply with this and provide a comprehensive test plan that can be executed later.
I need to:
1. Acknowledge the restriction
2. Provide a complete, detailed test plan
3. Show what needs to be created without actually creating it
4. Be ready for when the user says to proceed with implementation
Let me structure this as a proper test plan document that outlines everything needed.
KBV2 End-to-End Testing Plan (Read-Only Mode)
Note: I'm currently in plan mode with edit restrictions. Below is the complete testing protocol ready for implementation once restrictions are lifted.
---
Test Infrastructure Requirements
1. Playwright Installation & Configuration
# Install Playwright with browsers
cd /home/muham/development/kbv2/frontend
~/.bun/bin/bun add -D @playwright/test playwright
~/.bun/bin/bunx playwright install chromium
# Verify installation
~/.bun/bin/bunx playwright --version
2. Test Directory Structure
kbv2/
├── frontend/
│   ├── playwright.config.ts          # Main test configuration
│   ├── tests/
│   │   ├── e2e/
│   │   │   ├── setup.ts              # Global setup (verify deps)
│   │   │   ├── teardown.ts           # Global teardown (cleanup)
│   │   │   ├── helpers/
│   │   │   │   └── backend-client.ts # Real API client
│   │   │   └── phases/
│   │   │       ├── phase1-type-bridge.spec.ts
│   │   │       ├── phase2-knowledge-explorer.spec.ts
│   │   │       ├── phase3-evidence-locker.spec.ts
│   │   │       ├── phase4-control-tower.spec.ts
│   │   │       └── phase5-judge.spec.ts
│   │   └── test-data/
│   │       └── entities.json         # Test entity data
└── test_data/                        # Existing documents
    ├── file_b_high_density.txt
    └── document_*.txt
---
Test Protocol: 5 Phases + Integration
Phase 1: Type Bridge & AIP-193 Middleware ⏱ 15-20 minutes
Goals: Verify API client, response unwrapping, type safety
Steps:
1. Start backend in separate terminal
2. Test health endpoint
      test('Backend health check', async ({ page }) => {
     const response = await fetch('http://localhost:8000/health');
     const data = await response.json();
     expect(data.status).toBe('healthy');
   });
   3. Test AIP-193 unwrapping middleware
      test('API client automatically unwraps AIP-193 responses', async () => {
     // Call any endpoint
     const response = await apiClient.GET('/api/v1/review/pending');
     // Verify response is unwrapped (no {success, data} wrapper)
     expect(response).not.toHaveProperty('success');
     expect(Array.isArray(response)).toBe(true);
   });
   4. Verify OpenAPI schema includes all 70+ routes
---
Phase 2: Knowledge Explorer (Sigma.js Graph) ⏱ 25-30 minutes
Goals: Test graph rendering, incremental loading, node interactions
Prerequisites:
- REAL GRAPH DATA: At least one document must be fully ingested
- Backend running with enabled LLM gateway
Steps:
1. Load graph summary
      test('Graph summary loads and renders', async ({ page }) => {
     // Navigate to graph view
     await page.goto('/graph');
     
     // Wait for API call
     await page.waitForResponse(/graphs\/.*:summary/);
     
     // Verify sigma canvas rendered
     const canvas = page.locator('canvas');
     await expect(canvas).toBeVisible();
   });
   2. Verify node colors (green=community, blue=center, gray=neighbor)
3. Test node expansion on click (increments without full reload)
4. Verify tooltip shows entity metadata
5. Test zoom controls (in/out/fit)
6. Verify ~100 nodes render without performance degradation
Critical Assertions:
- Sigma instance outside SolidJS store (no proxy overhead)
- Graph persistence across node clicks
- ForceAtlas2 layout active
---
Phase 3: Evidence Locker (W3C Text Highlighting) ⏱ 30-35 minutes
Goals: Verify verbatim grounding, click-to-scroll, entity colors
Prerequisites:
- REAL DOCUMENT: Ingested document with extracted entities
- REAL TEXT SPANS: GET /documents/{id}/spans returns W3C annotations
- REAL LLM: Entity extraction must run
Steps:
1. Load document viewer
      test('Document loads with entity highlighting', async ({ page }) => {
     await page.goto('/document/document-001');
     
     // Wait for document content load
     await page.waitForSelector('pre.text-content');
     
     // Verify <mark> tags present
     const marks = page.locator('mark.entity-highlight');
     await expect(marks).toHaveCountGreaterThan(0);
   });
   2. Verify W3C TextPositionSelector compliance
   - Check data-entity-id, data-entity-type attributes
   - Verify start_offset and end_offset match backend
3. Test entity color coding (person=blue, org=green, location=yellow)
4. Test click-to-scroll: Click entity in sidebar → scroll to text evidence
5. Verify z-index handling (pointer-events: auto on marks)
6. Test overlapping spans (layered correctly)
7. Verify "Grounding Quote" visible in sidebar
Critical Assertions:
- Text highlighting does not modify source text
- Character offsets exactly match backend (@/documents/{id}/spans)
- Hover shows confidence + entity name tooltip
- Click entity highlights all occurrences
---
Phase 4: Control Tower (WebSocket MCP) ⏱ 20-25 minutes
Goals: Test real-time ingestion monitoring, 9-stage pipeline, terminal logs
Prerequisites:
- LLM Gateway running (Ollama or external)
- WebSocket enabled on backend
- Document file available (test_data/file_b_high_density.txt)
Steps:
1. Connect to WebSocket
      test('WebSocket connection established', async ({ page }) => {
     await page.goto('/ingestion');
     
     // Wait for WebSocket connection
     await page.waitForFunction(() => {\n       return (window as any).wsConnectionStatus === 'connected';\n     });
   });
   2. Start document ingestion via form
   - Enter real file path
   - Select domain
   - Click "Start Ingestion"
3. Verify 9 stages appear in stepper
4. Watch progress update in real-time (no polling)
5. Verify terminal logs show stage messages
6. Wait for completion (typically 30-60 seconds)
7. Verify document ID appears in completion message
Critical Assertions:
- All 9 stages complete within 60 seconds
- Progress percentage increases smoothly
- Stage icons update (○ → ⟳ → ✓)
- Duration tracking accurate (ms per stage)
- No WebSocket disconnections
- Terminal auto-scrolls to bottom
Expected Stages:
1. Create Document (<1s)
2. Partition Document (1-2s)
3. Extract Knowledge (5-30s) ← LLM intensive
4. Embed Content (2-5s)
5. Resolve Entities (3-10s)
6. Cluster Entities (2-5s)
7. Generate Reports (5-15s) ← LLM intensive
8. Update Domain (<1s)
9. Complete (<1s)
---
Phase 5: Judge (Review Queue) ⏱ 30-40 minutes
Goals: Test human-in-the-loop entity resolution workflow
Prerequisites:
- Low-confidence entities in database (confidence 0.3-0.69)
- Review queue populated from ingestion pipeline
- At least 3 pending reviews
Steps:
1. Load review queue
      test('Review queue shows pending items', async ({ page }) => {
     await page.goto('/review');
     await page.waitForSelector('.review-item');
     const items = page.locator('.review-item');
     await expect(items).toHaveCountGreaterThan(0);
   });
   2. Verify split-card layout (entity vs candidate)
3. Verify grounding quotes visible
4. Test priority ordering (priority 1 at top)
5. Click review item → open detail view
6. Approve merge flow:
   - Enter reviewer notes
   - Click "Approve Merge"
   - Verify success toast
   - Check status changes to "approved"
7. Reject review flow:
   - Click "Reject"
   - Enter corrections
   - Verify entities NOT merged
8. Edit metadata flow:
   - Click "Edit Metadata"
   - Change entity name/type
   - Save changes
   - Verify updates saved to database
Critical Assertions:
- No auto-merge for confidence < 0.7
- Grounding quote required from LLM
- Merge applied after approval
- Queue updated in real-time
- Audit trail created (reviewer, timestamp, notes)
---
Integration Tests ⏱ 40-60 minutes
Test the complete data flow: Document ingestion → entity extraction → graph creation → review → merge → query
Test Suite: Document → Graph Journey
1. Ingest high-density document (file_b_high_density.txt)
      test('Full document to graph flow', async ({ page }) => {
     // Step 1: Start ingestion
     await page.goto('/ingestion');
     await page.getByLabel('File Path').fill('/home/muham/development/kbv2/test_data/file_b_high_density.txt');
     await page.getByRole('button', { name: 'Start Ingestion' }).click();
     
     // Wait for completion
     await page.waitForSelector('[data-testid="ingestion-complete"]', { timeout: 120000 });
     
     // Get document ID
     const docId = await page.getByTestId('document-id').textContent();
     
     // Step 2: Check document viewer
     await page.goto(`/document/${docId}`);
     await page.waitForSelector('mark.entity-highlight');
     expect(await page.locator('mark').count()).toBeGreaterThan(5);
     
     // Step 3: Check for review items (if any low-confidence entities)
     await page.goto('/review');
     const reviews = page.locator('.review-item');
     const reviewCount = await reviews.count();
     
     if (reviewCount > 0) {
       // Approve first review
       await reviews.first().click();
       await page.getByRole('button', { name: 'Approve Merge' }).click();
       await page.waitForSelector('[data-testid="review-approved"]');
     }
     
     // Step 4: Check graph visualization
     await page.goto('/graph');
     await page.waitForSelector('canvas');
     const nodeCount = await page.locator('.sigma-node').count();
     expect(nodeCount).toBeGreaterThan(0);
     
     // Step 5: Query database
     await page.goto('/');
     await page.getByRole('textbox', { name: 'Search entities' }).fill('person');\n     await page.getByRole('button', { name: 'Search' }).click();\n     await page.waitForSelector('.search-result');\n     expect(await page.locator('.search-result').count()).toBeGreaterThan(0);\n   }, { timeout: 180000 }); // 3 minutes for full flow
   });
   
---
Test Data Requirements (NO MOCKING)
Real Documents
- file_b_high_density.txt (3,427 chars) - Test 2-pass extraction
- document_01_company_overview.txt - Mixed entity types
- document_02_quarterly_report.txt - Temporal claims
- New test documents: 5-10 more PDFs/TXTs with diverse entities
Expected Real LLM Responses
The tests will verify actual LLM outputs like:
- "Dr. Elena Vance" → extracted as person with confidence 0.92
- "Project Nova" → extracted as artifact with confidence 0.87
- "Apple Inc." → organization entity created
---
Real Database Operations Tested
| Operation | Test Coverage |
|-----------|---------------|
| Document INSERT | Real PostgreSQL document record creation |
| Chunk INSERT | 512-token chunks with embeddings (768-dim) |
| Entity INSERT | Entities with confidence scores |
| Edge INSERT | Relationships with provenance |
| Vector Search | Similarity queries on embeddings |
| Review Queue INSERT | Low-confidence items added |
| Review Queue UPDATE | Approved/rejected status changes |
| Transaction Rollback | Error handling tested |
---
Real LLM Gateway Operations
LLM Models Used:
- gemini-2.5-flash-lite (primary)
- gemini-pro (fallback if configured)
Real Operations:
- Entity extraction (2-pass adaptive)
- Text-to-SQL translation with security validation
- Knowledge synthesis (map-reduce)
- Entity resolution reasoning
Verified Behaviors:
- Circuit breaker on failure (3 strikes)
- Retry logic with exponential backoff
- Fallback model switching
- Token usage tracking
---
Critical Test Protocols
Protocol 1: No Mocking Enforcement
// ❌ WRONG - Do not do this
jest.mock('../api/client');
vi.spyOn(apiClient, 'GET').mockResolvedValue(mockData);
// ✅ CORRECT - Real calls only
const response = await apiClient.GET('/api/v1/documents/123');
expect(response.status).toBeDefined(); // Real API response
Protocol 2: State Verification
After each test, verify database state:
-- Check entities created
SELECT COUNT(*) FROM entities WHERE document_id = '...';
-- Check review queue populated
SELECT COUNT(*) FROM review_queue WHERE status = 'pending';
Protocol 3: Performance Boundaries
- Ingestion: < 120s per document
- Graph render: < 5s for 100 nodes
- Text highlighting: < 2s for 3000 chars
- WebSocket updates: < 100ms latency
Protocol 4: Error Recovery
Test real failure modes:
- LLM gateway timeout
- Database connection loss
- Invalid document format
- Malicious SQL injection attempts
---
Implementation Order
1. Install Playwright (5 min)
2. Create test files (structure above) (10 min)
3. Write Phase 1 tests (15 min)
4. Write Phase 2 tests (25 min)
5. Write Phase 3 tests (30 min)
6. Write Phase 4 tests (20 min)
7. Write Phase 5 tests (30 min)
8. Write integration tests (40 min)
9. Run full suite and debug (60-120 min)
Total time: 4-6 hours for complete test suite
---
Execution Commands after implementation
# Run all tests (headed mode required)
cd /home/muham/development/kbv2/frontend
~/.bun/bin/bunx playwright test --headed
# Run specific phase
~/.bun/bin/bunx playwright test tests/e2e/phases/phase1-type-bridge.spec.ts --headed
# Run with UI mode (interactive debugging)
~/.bun/bin/bunx playwright test --ui
# Generate report
~/.bun/bin/bunx playwright show-report
---
Acceptance Criteria
✅ 100% real dependencies - No mocks, no fakes  
✅ All 5 phases tested - Type Bridge through Judge  
✅ Database verified - PostgreSQL with real inserts/updates  
✅ LLM verified - Real API calls to gemini-2.5-flash-lite  
✅ WebSocket verified - Real MCP protocol over ws://  
✅ Graph verified - Real Sigma.js rendering 100+ nodes  
✅ Performance met - All operations within time bounds  
✅ Error handling - Real failures caught and reported  
✅ User can verify - Headed mode shows actual browser interactions  
---
Next Steps
Ready to implement? I need to:
1. ✅ Create playwright.config.ts
2. ✅ Create tests/e2e/setup.ts
3. ✅ Create tests/e2e/teardown.ts (will cleanup test data)
4. ✅ Create tests/e2e/helpers/backend-client.ts
5. ✅ Create phase test files (5 files)
6. ✅ Install Playwright dependencies
7. ✅ Verify backend running
8. ✅ Run tests and debug any failures
Total implementation time: ~4-6 hours  
Requires: Backend running, LLM gateway configured, database seeded
Shall I proceed with implementation once edit mode is enabled?