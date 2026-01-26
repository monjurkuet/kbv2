import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: './openapi-schema.json',
  output: {
    path: 'src/api',
    format: 'prettier'
  },
  client: '@hey-api/client-fetch',
  plugins: [
    '@hey-api/typescript',
    '@hey-api/sdk'
  ]
});

export {};
