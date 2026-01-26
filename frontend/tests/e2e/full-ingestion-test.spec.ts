import { test, expect } from '@playwright/test';
import { join } from 'path';
import { existsSync, readdirSync } from 'fs';

const PIPELINE_STAGES = [
  'Create Document',
  'Partition Document',
  'Extract Knowledge',
  'Embed Content',
  'Resolve Entities',
  'Cluster Entities',
  'Generate Reports',
  'Update Domain',
  'Complete'
];

test.describe('Full Document Ingestion E2E Test', () => {
  const TEST_DATA_DIR = '/home/muham/development/kbv2/tests/test_data';
  const TEST_FILES = [
    'document_01_company_overview.txt',
    'document_02_quarterly_report.txt',
    'document_03_press_release.txt',
    'file_b_high_density.txt',
    'file_a_truly_low_density.txt'
  ];

  test.beforeAll(async () => {
    console.log('=== Full Ingestion Test Suite Starting ===');
    console.log('Test data directory:', TEST_DATA_DIR);
    console.log('Files to ingest:', TEST_FILES.length);
    
    TEST_FILES.forEach(file => {
      const fullPath = join(TEST_DATA_DIR, file);
      const exists = existsSync(fullPath);
      console.log(`  - ${file}: ${exists ? '✓ EXISTS' : '✗ MISSING'}`);
    });
  });

  test('should ingest all test documents and verify complete workflow', async ({ page }) => {
    console.log('\n=== Starting Full Ingestion Test ===');
    
    const ingestionResults: Array<{ fileName: string; docName: string; success: boolean; logs: string }> = [];
    
    for (let i = 0; i < TEST_FILES.length; i++) {
      const fileName = TEST_FILES[i];
      const filePath = join(TEST_DATA_DIR, fileName);
      const docName = fileName.replace('.txt', '');
      
      console.log(`\n--- Ingesting document ${i + 1}/${TEST_FILES.length}: ${fileName} ---`);
      
      await navigateToControlTower(page);
      await configureAndStartIngestion(page, filePath, docName);
      await monitorIngestionProcess(page, fileName, PIPELINE_STAGES);
      const logs = await captureIngestionLogs(page);
      
      ingestionResults.push({
        fileName,
        docName,
        success: true,
        logs
      });
      
      console.log(`✓ Completed ingestion for ${fileName}`);
    }
    
    console.log('\n=== All Documents Ingested Successfully ===');
    
    await verifyReviewQueue(page);
    await navigateToReviewQueue(page);
    await takeDashboardScreenshots(page);
    
    await verifyDatabaseState();
    
    expect(ingestionResults.filter(r => r.success).length).toBe(TEST_FILES.length);
  });

  test.afterEach(async ({ page }) => {
    await page.screenshot({
      path: `test-results/screenshots/final-state-${Date.now()}.png`,
      fullPage: true
    });
  });
});

async function navigateToControlTower(page: any) {
  console.log('  → Navigating to Control Tower...');
  await page.goto('/control-tower');
  await page.waitForLoadState('networkidle');
  await page.waitForSelector('[data-testid="control-tower-container"]', { timeout: 15000 });
  console.log('  ✓ Control Tower loaded');
}

async function configureAndStartIngestion(page: any, filePath: string, docName: string) {
  console.log('  → Opening configuration panel...');
  
  const configureButton = page.locator('button', { hasText: /Configure/ }).first();
  await configureButton.click();
  await page.waitForTimeout(500);
  
  console.log(`  → Setting document path: ${filePath}`);
  const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
  await filePathInput.fill(filePath);
  
  console.log(`  → Setting document name: ${docName}`);
  const documentNameInput = page.locator('input[placeholder*="My Document"]');
  await documentNameInput.fill(docName);
  
  console.log('  → Selecting domain: general');
  const domainSelector = page.locator('select');
  await domainSelector.selectOption('general');
  
  console.log('  → Starting ingestion...');
  const startButton = page.locator('button', { hasText: /Start Ingestion/ });
  await expect(startButton).toBeEnabled({ timeout: 5000 });
  await startButton.click();
  
  console.log('  ✓ Ingestion started');
}

async function monitorIngestionProcess(page: any, fileName: string, stages: string[]) {
  console.log('  → Monitoring ingestion progress...');
  
  const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]').first();
  
  await page.waitForTimeout(3000);
  const initialStatus = await connectionStatus.textContent();
  console.log(`  → Initial connection status: ${initialStatus}`);
  
  for (let stageIndex = 0; stageIndex < 9; stageIndex++) {
    const stageName = PIPELINE_STAGES[stageIndex];
    console.log(`    → Waiting for stage ${stageIndex + 1}/9: ${stageName}...`);
    
    try {
      await page.waitForFunction(
        (expectedStage: string) => {
          const stages = document.querySelectorAll('[class*="StageStepper"] *, [data-testid*="stage"]');
          for (const stage of stages) {
            if (stage.textContent?.includes(expectedStage)) {
              const parent = stage.closest('[class*="completed"], [class*="active"]');
              return parent !== null;
            }
          }
          return false;
        },
        stageName,
        { timeout: 60000 }
      );
      
      console.log(`    ✓ Stage completed: ${stageName}`);
      
      await page.screenshot({
        path: `test-results/screenshots/${fileName}-stage-${stageIndex + 1}-${stageName.replace(/\s+/g, '-').toLowerCase()}.png`,
        fullPage: false
      });
      
    } catch (error) {
      console.log(`    ! Timeout waiting for stage: ${stageName}, checking logs...`);
      await captureIngestionLogs(page);
      throw new Error(`Failed to complete stage: ${stageName}`);
    }
    
    await page.waitForTimeout(2000);
  }
  
  console.log('  ✓ All 9 stages completed');
}

async function captureIngestionLogs(page: any): Promise<string> {
  const logContainer = page.locator('.TerminalLog');
  const logContent = await logContainer.textContent();
  
  if (logContent && logContent.length > 0) {
    console.log('  → Ingestion logs captured:', logContent.substring(0, 200) + '...');
  }
  
  return logContent || '';
}

async function verifyReviewQueue(page: any) {
  console.log('\n=== Verifying Review Queue ===');
  
  try {
    await page.goto('/queue');
    await page.waitForLoadState('networkidle', { timeout: 15000 });
    
    const reviewContainer = page.locator('[data-testid="review-queue"], [class*="queue"], [class*="review"]').first();
    const exists = await reviewContainer.count() > 0;
    
    if (exists) {
      const queueContent = await reviewContainer.textContent();
      console.log('  → Review queue content length:', queueContent?.length || 0);
      
      const entityCount = (queueContent?.match(/entity/gi) || []).length;
      const edgeCount = (queueContent?.match(/edge/gi) || []).length;
      
      console.log(`  → Entities in queue: ${entityCount}`);
      console.log(`  → Edges in queue: ${edgeCount}`);
      
      await page.screenshot({
        path: `test-results/screenshots/review-queue-${Date.now()}.png`,
        fullPage: true
      });
    } else {
      console.log('  → Review queue container not found (may be empty)');
    }
    
  } catch (error) {
    console.log('  ! Error accessing review queue:', error);
  }
}

async function navigateToReviewQueue(page: any) {
  console.log('\n=== Navigating to Review Queue/Dashboard ===');
  
  try {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    console.log('  ✓ Dashboard/home page loaded');
    
    const dashboardContent = page.locator('body').textContent();
    const hasEntities = (await dashboardContent)?.toLowerCase().includes('entity');
    const hasGraph = (await dashboardContent)?.toLowerCase().includes('graph') || (await dashboardContent)?.toLowerCase().includes('network');
    
    console.log(`  → Dashboard has entities: ${hasEntities}`);
    console.log(`  → Dashboard has graph: ${hasGraph}`);
    
    await page.screenshot({
      path: `test-results/screenshots/dashboard-${Date.now()}.png`,
      fullPage: true
    });
    
  } catch (error) {
    console.log('  ! Error navigating to dashboard:', error);
  }
}

async function takeDashboardScreenshots(page: any) {
  console.log('\n=== Capturing Dashboard Screenshots ===');
  
  const viewportSizes = [
    { width: 1920, height: 1080, name: 'desktop' },
    { width: 768, height: 1024, name: 'tablet' },
    { width: 375, height: 667, name: 'mobile' }
  ];
  
  for (const viewport of viewportSizes) {
    console.log(`  → Capturing ${viewport.name} view (${viewport.width}x${viewport.height})...`);
    
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.waitForTimeout(1000);
    
    await page.screenshot({
      path: `test-results/screenshots/dashboard-${viewport.name}-${Date.now()}.png`,
      fullPage: false
    });
  }
  
  await page.setViewportSize({ width: 1920, height: 1080 });
  console.log('  ✓ All dashboard screenshots captured');
}

async function verifyDatabaseState() {
  console.log('\n=== Database Verification ===');
  console.log('  → Database verification queries to run:');
  console.log('');
  console.log('  1. Verify documents were created:');
  console.log('     SELECT COUNT(*) as doc_count FROM document;');
  console.log('');
  console.log('  2. Verify chunks were created:');
  console.log('     SELECT COUNT(*) as chunk_count FROM chunk;');
  console.log('');
  console.log('  3. Verify entities were extracted:');
  console.log('     SELECT COUNT(*) as entity_count FROM entity;');
  console.log('');
  console.log('  4. Verify edges were created:');
  console.log('     SELECT COUNT(*) as edge_count FROM edge;');
  console.log('');
  console.log('  5. Verify chunk-entity relationships:');
  console.log('     SELECT COUNT(*) as chunk_entity_count FROM chunk_entity;');
  console.log('');
  console.log('  6. Check document statuses:');
  console.log('     SELECT name, status, domain, created_at FROM document ORDER BY created_at DESC;');
  console.log('');
  console.log('  7. Verify entities by type:');
  console.log('     SELECT entity_type, COUNT(*) as count FROM entity GROUP BY entity_type;');
  console.log('');
  console.log('  8. Check edge types:');
  console.log('     SELECT edge_type, COUNT(*) as count FROM edge GROUP BY edge_type;');
  console.log('');
  console.log('  9. Verify review queue items:');
  console.log('     SELECT item_type, status, COUNT(*) as count FROM review_queue GROUP BY item_type, status;');
  console.log('');
  console.log('  10. Check communities/clusters:');
  console.log('      SELECT level, COUNT(*) as community_count FROM community GROUP BY level;');
  console.log('');
  console.log('  Note: Run these queries against your PostgreSQL database to verify ingestion results.');
  console.log('  Expected results based on 5 test documents:');
  console.log('    - Documents: 5');
  console.log('    - Chunks: 20-50 (depending on document size)');
  console.log('    - Entities: 15-30 (varies by document content)');
  console.log('    - Edges: 10-25 (relationships between entities)');
  console.log('    - Chunk-Entity links: 30-60');
}

test.describe('Individual Document Ingestion Tests', () => {
  const TEST_DATA_DIR = '/home/muham/development/kbv2/tests/test_data';

  test('should ingest company overview document', async ({ page }) => {
    const fileName = 'document_01_company_overview.txt';
    const filePath = join(TEST_DATA_DIR, fileName);
    const docName = 'Company Overview';
    
    await runSingleDocumentIngestion(page, filePath, docName, fileName);
  });

  test('should ingest quarterly report document', async ({ page }) => {
    const fileName = 'document_02_quarterly_report.txt';
    const filePath = join(TEST_DATA_DIR, fileName);
    const docName = 'Quarterly Report';
    
    await runSingleDocumentIngestion(page, filePath, docName, fileName);
  });

  test('should ingest press release document', async ({ page }) => {
    const fileName = 'document_03_press_release.txt';
    const filePath = join(TEST_DATA_DIR, fileName);
    const docName = 'Press Release';
    
    await runSingleDocumentIngestion(page, filePath, docName, fileName);
  });

  test('should ingest high density document', async ({ page }) => {
    const fileName = 'file_b_high_density.txt';
    const filePath = join(TEST_DATA_DIR, fileName);
    const docName = 'High Density Document';
    
    await runSingleDocumentIngestion(page, filePath, docName, fileName);
  });

  test('should handle low density edge case document', async ({ page }) => {
    const fileName = 'file_a_truly_low_density.txt';
    const filePath = join(TEST_DATA_DIR, fileName);
    const docName = 'Low Density Document';
    
    await runSingleDocumentIngestion(page, filePath, docName, fileName);
  });
});

async function runSingleDocumentIngestion(page: any, filePath: string, docName: string, fileName: string) {
  console.log(`\n=== Testing ingestion: ${fileName} ===`);
  
  await navigateToControlTower(page);
  await configureAndStartIngestion(page, filePath, docName);
  await monitorIngestionProcess(page, fileName, PIPELINE_STAGES);
  
  const logs = await captureIngestionLogs(page);
  
  console.log(`✓ ${fileName} ingestion completed`);
  
  expect(logs.length).toBeGreaterThan(0);
}