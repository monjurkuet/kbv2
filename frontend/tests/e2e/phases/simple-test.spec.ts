import { test, expect } from '@playwright/test';

test.describe('Phase 2: Knowledge Explorer', () => {
  test('graph page loads', async ({ page }) => {
    await page.goto('/graph');
    
    await page.waitForLoadState('networkidle');
    
    const title = await page.title();
    console.log('Page title:', title);
    
    const bodyText = await page.textContent('body');
    console.log('Body text length:', bodyText?.length);
    
    const hasRoot = await page.locator('#root').count();
    console.log('Root element count:', hasRoot);
    
    const rootHTML = await page.locator('#root').innerHTML();
    console.log('Root HTML length:', rootHTML.length);
    
    expect(hasRoot).toBeGreaterThan(0);
  });
});