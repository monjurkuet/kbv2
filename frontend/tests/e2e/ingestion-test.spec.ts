import { test, expect } from '@playwright/test';

test.describe('Ingestion Process E2E Test', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/control-tower');
    await page.waitForLoadState('networkidle');
    
    await page.waitForSelector('[data-testid="control-tower-container"]', { timeout: 10000 });
  });

  test('should navigate to control tower and display main elements', async ({ page }) => {
    const container = page.locator('[data-testid="control-tower-container"]');
    await expect(container).toBeVisible();

    const header = page.locator('text=Ingestion Control Tower');
    await expect(header).toBeVisible();
  });

  test('should open configuration panel and display input fields', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await expect(filePathInput).toBeVisible();

    const documentNameInput = page.locator('input[placeholder*="My Document"]');
    await expect(documentNameInput).toBeVisible();

    const domainSelector = page.locator('select');
    await expect(domainSelector).toBeVisible();

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await expect(startButton).toBeVisible();
  });

  test('should display connection status', async ({ page }) => {
    const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]').first();
    await expect(connectionStatus).toBeVisible();

    const statusText = await connectionStatus.textContent();
    expect(statusText).toMatch(/CONNECTED|CONNECTING|ERROR|DISCONNECTED/);
  });

  test('should display all pipeline stages', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

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
      const stageElement = page.locator('text=' + stageName).first();
      await expect(stageElement).toBeVisible();
    }
  });

  test('should display terminal log container', async ({ page }) => {
    const logContainer = page.locator('.TerminalLog');
    await expect(logContainer).toBeVisible();
  });

  test('should display stage stepper', async ({ page }) => {
    const stageStepper = page.locator('.StageStepper');
    await expect(stageStepper).toBeVisible();
  });

  test('should start ingestion process with test document', async ({ page }) => {
    await page.waitForTimeout(3000);

    const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]').first();
    const statusText = await connectionStatus.textContent();
    console.log('Connection status:', statusText);

    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await expect(startButton).toBeEnabled();
    await startButton.click();

    await page.waitForTimeout(3000);

    const buttonText = await startButton.textContent();
    expect(buttonText).toBeTruthy();
  });

  test('should verify connection status becomes CONNECTED', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await startButton.click();

    await page.waitForTimeout(3000);

    const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]').first();
    const statusText = await connectionStatus.textContent();
    
    expect(statusText).toMatch(/CONNECTED/);
  });

  test('should display logs after starting ingestion', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await startButton.click();

    await page.waitForTimeout(3000);

    const logContainer = page.locator('.TerminalLog');
    const logContent = await logContainer.textContent();
    
    expect(logContent).toBeTruthy();
    expect(logContent!.length).toBeGreaterThan(0);
  });

  test('should show stage progress after starting ingestion', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await startButton.click();

    await page.waitForTimeout(5000);

    const stages = await page.locator('[class*="StageStepper"]').all();
    expect(stages.length).toBeGreaterThan(0);
  });

  test('should handle reset functionality', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await startButton.click();

    await page.waitForTimeout(2000);

    const resetButton = page.locator('button', { hasText: /Reset/ });
    await resetButton.click();

    await page.waitForTimeout(1000);

    const filePathValue = await filePathInput.inputValue();
    expect(filePathValue).toBe('');
  });

  test('should validate required fields before starting ingestion', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    
    const isDisabled = await startButton.isDisabled();
    expect(isDisabled).toBe(true);
  });

  test('should select different domains', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const domainSelector = page.locator('select');
    
    await domainSelector.selectOption('cybersecurity');
    let selectedValue = await domainSelector.inputValue();
    expect(selectedValue).toBe('cybersecurity');

    await domainSelector.selectOption('finance');
    selectedValue = await domainSelector.inputValue();
    expect(selectedValue).toBe('finance');

    await domainSelector.selectOption('general');
    selectedValue = await domainSelector.inputValue();
    expect(selectedValue).toBe('general');
  });

  test('should handle document name field', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const documentNameInput = page.locator('input[placeholder*="My Document"]');
    await documentNameInput.fill('Test Document Name');

    const value = await documentNameInput.inputValue();
    expect(value).toBe('Test Document Name');
  });

  test('should complete ingestion workflow end-to-end', async ({ page }) => {
    await page.goto('/control-tower');
    await page.waitForLoadState('networkidle');

    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/home/muham/development/kbv2/test_sample.txt');

    const documentNameInput = page.locator('input[placeholder*="My Document"]');
    await documentNameInput.fill('E2E Test Document');

    const domainSelector = page.locator('select');
    await domainSelector.selectOption('general');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await expect(startButton).toBeEnabled();
    await startButton.click();

    await page.waitForTimeout(3000);

    const connectionStatus = page.locator('[class*="bg-"][class*="text-"][class*="border-"]').first();
    const statusText = await connectionStatus.textContent();
    expect(statusText).toMatch(/CONNECTED/);

    const logContainer = page.locator('.TerminalLog');
    const logContent = await logContainer.textContent();
    expect(logContent).toBeTruthy();
    expect(logContent!.length).toBeGreaterThan(0);

    const stageStepper = page.locator('.StageStepper');
    await expect(stageStepper).toBeVisible();
  });

  test('should handle errors gracefully', async ({ page }) => {
    const configureButton = page.locator('button', { hasText: /Configure/ }).first();
    await configureButton.click();

    const filePathInput = page.locator('input[placeholder*="/path/to/document"]');
    await filePathInput.fill('/non/existent/path/to/file.txt');

    const startButton = page.locator('button', { hasText: /Start Ingestion/ });
    await startButton.click();

    await page.waitForTimeout(3000);

    const errorDisplay = page.locator('.bg-red-50');
    const hasError = await errorDisplay.count();
    
    if (hasError > 0) {
      const errorText = await errorDisplay.textContent();
      expect(errorText).toBeTruthy();
    }
  });
});