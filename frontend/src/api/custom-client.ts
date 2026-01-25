export const apiClient = {
  GET: async (url: string, options?: { params?: any }) => {
    const fullUrl = new URL(`http://localhost:5001${url}`);
    
    if (options?.params?.path) {
      Object.entries(options.params.path).forEach(([key, value]) => {
        fullUrl.pathname = fullUrl.pathname.replace(`{${key}}`, String(value));
      });
    }
    
    if (options?.params?.query) {
      Object.entries(options.params.query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          fullUrl.searchParams.append(key, String(value));
        }
      });
    }
    
    const response = await fetch(fullUrl.toString(), {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const json = await response.json();
    
    // Unwrap AIP-193 response format
    if (json && typeof json === 'object' && 'success' in json) {
      if (json.success === false && json.error) {
        throw new Error(json.error.message || 'API request failed');
      }
      return json.data;
    }
    
    return json;
  },
  POST: async (url: string, options?: { body?: any; params?: any }) => {
    const fullUrl = new URL(`http://localhost:5001${url}`);
    
    if (options?.params?.path) {
      Object.entries(options.params.path).forEach(([key, value]) => {
        fullUrl.pathname = fullUrl.pathname.replace(`{${key}}`, String(value));
      });
    }
    
    if (options?.params?.query) {
      Object.entries(options.params.query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          fullUrl.searchParams.append(key, String(value));
        }
      });
    }
    
    const response = await fetch(fullUrl.toString(), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: options?.body ? JSON.stringify(options.body) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const json = await response.json();
    
    // Unwrap AIP-193 response format
    if (json && typeof json === 'object' && 'success' in json) {
      if (json.success === false && json.error) {
        throw new Error(json.error.message || 'API request failed');
      }
      return json.data;
    }
    
    return json;
  },
  PUT: async (url: string, options?: { body?: any; params?: any }) => {
    const fullUrl = new URL(`http://localhost:5001${url}`);
    
    if (options?.params?.path) {
      Object.entries(options.params.path).forEach(([key, value]) => {
        fullUrl.pathname = fullUrl.pathname.replace(`{${key}}`, String(value));
      });
    }
    
    if (options?.params?.query) {
      Object.entries(options.params.query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          fullUrl.searchParams.append(key, String(value));
        }
      });
    }
    
    const response = await fetch(fullUrl.toString(), {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: options?.body ? JSON.stringify(options.body) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const json = await response.json();
    
    // Unwrap AIP-193 response format
    if (json && typeof json === 'object' && 'success' in json) {
      if (json.success === false && json.error) {
        throw new Error(json.error.message || 'API request failed');
      }
      return json.data;
    }
    
    return json;
  },
  DELETE: async (url: string, options?: { params?: any }) => {
    const fullUrl = new URL(`http://localhost:5001${url}`);
    
    if (options?.params?.path) {
      Object.entries(options.params.path).forEach(([key, value]) => {
        fullUrl.pathname = fullUrl.pathname.replace(`{${key}}`, String(value));
      });
    }
    
    if (options?.params?.query) {
      Object.entries(options.params.query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          fullUrl.searchParams.append(key, String(value));
        }
      });
    }
    
    const response = await fetch(fullUrl.toString(), {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  },
  PATCH: async (url: string, options?: { body?: any; params?: any }) => {
    const fullUrl = new URL(`http://localhost:5001${url}`);
    
    if (options?.params?.path) {
      Object.entries(options.params.path).forEach(([key, value]) => {
        fullUrl.pathname = fullUrl.pathname.replace(`{${key}}`, String(value));
      });
    }
    
    if (options?.params?.query) {
      Object.entries(options.params.query).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          fullUrl.searchParams.append(key, String(value));
        }
      });
    }
    
    const response = await fetch(fullUrl.toString(), {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: options?.body ? JSON.stringify(options.body) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  },
};