import { test, expect } from '@playwright/test';

test.describe('Phase 5: Judge (Review Queue)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/review-queue');
    await page.waitForSelector('[data-testid="review-container"]');
  });

  test('Review queue loads and renders container', async ({ page }) => {
    const queueContainer = page.locator('[data-testid="review-container"]');
    await expect(queueContainer).toBeVisible();

    const reviewQueue = page.locator('.ReviewQueue');
    await expect(reviewQueue).toBeVisible();
  });

  test('Review queue header is visible', async ({ page }) => {
    const header = page.locator('text=Review Queue').first();
    await expect(header).toBeVisible();
  });

  test('Review items or empty state are displayed', async ({ page }) => {
    const reviewItems = page.locator('.ReviewQueue [class*="hover:bg-gray-"]');

    const itemCount = await reviewItems.count();
    if (itemCount === 0) {
      await expect(page.locator('text=/No pending reviews|All reviews have been processed/').first()).toBeVisible();
    } else {
      expect(itemCount).toBeGreaterThan(0);
      await expect(reviewItems.first()).toBeVisible();
    }
  });

  test('Review queue displays correct heading', async ({ page }) => {
    const heading = page.locator('h2:has-text("Review Queue")');
    await expect(heading).toBeVisible();
    
    const subheading = page.locator('text=Human-in-the-loop entity resolution');
    await expect(subheading).toBeVisible();
  });

  test('Review queue can be accessed from multiple routes', async ({ page }) => {
    const testRoutes = ['/', '/queue', '/review', '/review-queue'];
    
    for (const route of testRoutes) {
      await page.goto(route);
      await page.waitForSelector('[data-testid="review-container"]');
      const container = page.locator('[data-testid="review-container"]');
      await expect(container).toBeVisible();
    }
  });

  test('Review queue container has correct data-testid', async ({ page }) => {
    const container = page.locator('[data-testid="review-container"]');
    await expect(container).toBeVisible();
    await expect(container).toHaveAttribute('data-testid', 'review-container');
  });

  test('Review queue renders without errors', async ({ page }) => {
    const container = page.locator('[data-testid="review-container"]');
    
    const hasError = await container.evaluate(el => {
      return el.querySelector('[class*="error"]') !== null;
    });
    
    expect(hasError).toBe(false);
  });

  test('Review queue is responsive', async ({ page }) => {
    const container = page.locator('[data-testid="review-container"]');
    
    await page.setViewportSize({ width: 1920, height: 1080 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(container).toBeVisible();
    
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(container).toBeVisible();
  });
});