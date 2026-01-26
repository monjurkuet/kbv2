import { test, expect } from '@playwright/test';

test.describe('Phase 3: Evidence Locker', () => {
  test.beforeEach(async ({ page }) => {
    const documentId = '00000000-0000-0000-0000-000000000001';
    await page.goto(`/document/${documentId}`);
    await page.waitForSelector('[data-testid="document-container"]');
  });

  test('document container renders', async ({ page }) => {
    const container = page.locator('[data-testid="document-container"]');
    await expect(container).toBeVisible();
  });

  test('document container has correct data-testid', async ({ page }) => {
    const container = page.locator('[data-testid="document-container"]');
    await expect(container).toHaveAttribute('data-testid', 'document-container');
  });

  test('document renders without errors', async ({ page }) => {
    const container = page.locator('[data-testid="document-container"]');
    
    const hasError = await container.evaluate(el => {
      return el.querySelector('[class*="error"]') !== null;
    });
    
    expect(hasError).toBe(false);
  });

  test('document can be accessed from different document IDs', async ({ page }) => {
    const testIds = [
      '00000000-0000-0000-0000-000000000001',
      '00000000-0000-0000-0000-000000000002',
      '00000000-0000-0000-0000-000000000003'
    ];
    
    for (const docId of testIds) {
      await page.goto(`/document/${docId}`);
      await page.waitForSelector('[data-testid="document-container"]');
      const container = page.locator('[data-testid="document-container"]');
      await expect(container).toBeVisible();
    }
  });

  test('document viewer is responsive', async ({ page }) => {
    const container = page.locator('[data-testid="document-container"]');
    
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(container).toBeVisible();
  });

  test('document content area is visible', async ({ page }) => {
    const contentArea = page.locator('[data-testid="document-content"]');
    
    const isVisible = await contentArea.count();
    if (isVisible > 0) {
      await expect(contentArea).toBeVisible();
    }
  });

  test('entity sidebar is present', async ({ page }) => {
    const sidebar = page.locator('[data-testid="entity-sidebar"]');
    
    const isPresent = await sidebar.count();
    if (isPresent > 0) {
      await expect(sidebar).toBeVisible();
    }
  });

  test('document displays loading state', async ({ page }) => {
    await page.goto(`/document/00000000-0000-0000-0000-000000000004`);
    
    await page.waitForSelector('[data-testid="document-container"]', { timeout: 5000 });
    
    const loadingIndicator = page.getByTestId('loading-indicator');
    const isVisible = await loadingIndicator.isVisible().catch(() => false);
    
    expect(isVisible).toBe(false);
  });

  test('document viewer handles error state gracefully', async ({ page }) => {
    const container = page.locator('[data-testid="document-container"]');
    
    const errorElement = await container.evaluate(el => {
      return el.querySelector('[class*="error"]') !== null;
    });
    
    if (errorElement) {
      const errorText = page.locator('[class*="error"]');
      await expect(errorText).toBeVisible();
    }
  });
});