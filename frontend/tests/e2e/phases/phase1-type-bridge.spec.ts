import { test, expect } from '@playwright/test';
import { backendClient } from '../helpers/backend-client';

test.describe('Phase 1: Type Bridge', () => {
  test('backend health check responds with 200', async ({ request }) => {
    const response = await request.get(`${process.env.API_BASE_URL || 'http://localhost:8765'}/health`);
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toMatchObject({
      status: expect.any(String),
      version: expect.any(String),
      name: expect.any(String)
    });
  });

  test('AIP-193 response unwrapping works correctly', async ({ request }) => {
    const response = await request.get(`${process.env.API_BASE_URL || 'http://localhost:8765'}/api/v1/openapi`);
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    expect(body).toHaveProperty('success');
    expect(body).toHaveProperty('data');
    expect(body).toHaveProperty('error');
    expect(body).toHaveProperty('metadata');
    
    expect(body.success).toBe(true);
    expect(body.data).toHaveProperty('openapi', '3.1.0');
    expect(body.data).toHaveProperty('info');
    expect(body.data).toHaveProperty('paths');
    expect(body.data).toHaveProperty('components');
  });

  test('OpenAPI schema validation', async ({ request }) => {
    const response = await request.get(`${process.env.API_BASE_URL || 'http://localhost:8765'}/api/v1/openapi`);
    expect(response.ok()).toBeTruthy();
    
    const json = await response.json();
    expect(json).toHaveProperty('success', true);
    expect(json).toHaveProperty('data');
    
    const schema = json.data;
    expect(schema).toHaveProperty('openapi');
    expect(schema).toHaveProperty('info');
    expect(schema).toHaveProperty('paths');
    expect(schema).toHaveProperty('components');
    
    expect(schema.info).toHaveProperty('title');
    expect(schema.info).toHaveProperty('version', '1.0.0');
    
    expect(schema.paths).toHaveProperty('/health');
    expect(schema.paths).toHaveProperty('/ready');
    expect(schema.paths).toHaveProperty('/api/v1/openapi');
    
    expect(schema.paths).toHaveProperty('/api/v1/query/translate');
    expect(schema.paths['/api/v1/query/translate'].post).toBeDefined();
    
    expect(schema.paths).toHaveProperty('/api/v1/review/pending');
    expect(schema.paths['/api/v1/review/pending'].get).toBeDefined();
    
    expect(schema.paths).toHaveProperty('/api/v1/graphs/{graph_id}:summary');
    expect(schema.paths['/api/v1/graphs/{graph_id}:summary'].get).toBeDefined();
    
    expect(schema.paths).toHaveProperty('/api/v1/documents/{document_id}');
    expect(schema.paths['/api/v1/documents/{document_id}'].get).toBeDefined();
  });
});