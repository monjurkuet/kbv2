interface AIP193Response<T = any> {
  success: boolean;
  data: T;
  error: {};
  metadata: {};
}

class BackendClient {
  private baseURL: string;
  
  constructor() {
    this.baseURL = process.env.API_BASE_URL || 'http://localhost:8765';
  }
  
  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return response.json();
  }
  
  async processEntity(text: string) {
    const response = await fetch(`${this.baseURL}/api/v1/query/api/v1/query/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ nl_query: text })
    });
    
    if (!response.ok) {
      throw new Error(`Entity processing failed: ${response.status}`);
    }
    
    const result: AIP193Response = await response.json();
    if (!result.success) {
      throw new Error(`Entity processing failed: ${JSON.stringify(result.error)}`);
    }
    
    return result;
  }
  
  async getOpenAPIData() {
    const response = await fetch(`${this.baseURL}/api/v1/openapi`);
    if (!response.ok) {
      throw new Error(`OpenAPI schema fetch failed: ${response.status}`);
    }
    const result: AIP193Response = await response.json();
    if (!result.success) {
      throw new Error(`OpenAPI schema fetch failed: ${JSON.stringify(result.error)}`);
    }
    return result;
  }
}

export const backendClient = new BackendClient();