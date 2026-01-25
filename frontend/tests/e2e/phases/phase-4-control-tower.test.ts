import { test, expect } from '@playwright/test';

test.describe('Phase 4: Control Tower (WebSocket MCP)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/control-tower');
    await page.waitForSelector('[data-testid="control-tower-container"]');
  });

  test('Control tower renders with UI elements', async ({ page }) => {
    const container = page.locator('[data-testid="control-tower-container"]');
    await expect(container).toBeVisible();
    
    const header = page.locator('text=Ingestion Control Tower');
    await expect(header).toBeVisible();
  });

  test('Connection status indicator is visible', async ({ page }) => {
    const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]');
    await expect(connectionStatus).toBeVisible();
    
    const statusText = await connectionStatus.textContent();
    expect(statusText).toMatch(/CONNECTED|CONNECTING|ERROR|DISCONNECTED/);
  });

  test('Configure button is present', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ });
    await expect(configureButton).toBeVisible();
  });

  test('Configuration panel can be toggled', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ });
    await configureButton.click();
    
    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await expect(filePathInput).toBeVisible();
  });

  test('9-stage pipeline stages are displayed', async ({ page }) => {
    const stages = [
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
    
    for (const stageName of stages) {
      const stageElement = page.locator('text=' + stageName);
      await expect(stageElement).toBeVisible();
    }
  });

  test('Terminal log container is visible', async ({ page }) => {
    const logContainer = page.locator('.TerminalLog');
    await expect(logContainer).toBeVisible();
  });

  test('Stage stepper is visible', async ({ page }) => {
    const stageStepper = page.locator('.StageStepper');
    await expect(stageStepper).toBeVisible();
  });

  test('Reset button is present', async ({ page }) => {
    const resetButton = page.locator('button', { hasText: /Reset/ });
    await expect(resetButton).toBeVisible();
  });

  test('File input fields are present in configuration panel', async ({ page }) => {
    await page.click('button', { hasText: /Configure/ });
    
    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await expect(filePathInput).toBeVisible();
    
    const documentNameInput = page.locator('input[placeholder*="My Document"]');
    await expect(documentNameInput).toBeVisible();
  });

  test('Domain selector is present', async ({ page }) => {
    await page.click('button', { hasText: /Configure/ });
    
    const domainSelector = page.locator('select');
    await expect(domainSelector).toBeVisible();
  });

  test('Start Ingestion button is present', async ({ page }) => {
    await page.click('button', { hasText: /Configure/ });
    
    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await expect(startButton).toBeVisible();
  });
});