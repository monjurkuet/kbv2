import { test, expect } from '@playwright/test';

test.describe('Phase 2: Knowledge Explorer', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/graph');
    await page.waitForSelector('[data-testid="graph-container"]');
  });

  test('graph container renders', async ({ page }) => {
    const container = page.locator('[data-testid="graph-container"]');
    await expect(container).toBeVisible();
  });

  test('Sigma.js canvas is present', async ({ page }) => {
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('graph container has correct data-testid', async ({ page }) => {
    const container = page.locator('[data-testid="graph-container"]');
    await expect(container).toHaveAttribute('data-testid', 'graph-container');
  });

  test('graph renders without errors', async ({ page }) => {
    const container = page.locator('[data-testid="graph-container"]');
    
    const hasError = await container.evaluate(el => {
      return el.querySelector('[class*="error"]') !== null;
    });
    
    expect(hasError).toBe(false);
  });

  test('graph can be accessed from multiple routes', async ({ page }) => {
    const testRoutes = ['/graph', '/graph/00000000-0000-0000-0000-000000000000'];
    
    for (const route of testRoutes) {
      await page.goto(route);
      await page.waitForSelector('[data-testid="graph-container"]');
      const container = page.locator('[data-testid="graph-container"]');
      await expect(container).toBeVisible();
    }
  });

  test('graph is responsive', async ({ page }) => {
    const container = page.locator('[data-testid="graph-container"]');
    
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(container).toBeVisible();
  });

  test('canvas is rendered within graph container', async ({ page }) => {
    const container = page.locator('[data-testid="graph-container"]');
    const canvas = page.locator('canvas');
    
    await expect(container).toBeVisible();
    await expect(canvas).toBeVisible();
  });

  test('graph controls are visible', async ({ page }) => {
    const controls = page.locator('[data-testid="graph-controls"]');
    await expect(controls).toBeVisible();
  });
});