export default async function globalSetup() {
  console.log('Setting up global test environment...');
  
  const backendUrl = process.env.API_BASE_URL || 'http://localhost:5001';
  
  try {
    const response = await fetch(`${backendUrl}/health`);
    if (!response.ok) {
      throw new Error(`Backend not healthy: ${response.status}`);
    }
    
    const health = await response.json();
    console.log(`Backend ready: ${health.name} v${health.version}`);
  } catch (error) {
    console.error('Failed to connect to backend:', error);
    throw new Error('Backend is not available. Please start the backend server first.');
  }
  
  console.log('Global setup complete');
}