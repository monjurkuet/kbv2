import { defineConfig } from '@hey-api/openapi-ts';

export default defineConfig({
  input: './openapi-schema.json',
  output: {
    path: 'src/api',
    format: 'prettier'
  },
  plugins: [
    {
      name: '@hey-api/typescript',
      config: {
        enums: 'typescript',
        dates: true,
        name: 'ApiTypes'
      }
    }
  ]
});

export {};
