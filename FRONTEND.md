# KBV2 Frontend Guide

## Overview

The KBV2 frontend is a **SolidJS-based TypeScript web application** for visualizing and managing knowledge graphs extracted from documents. It provides a rich interface for document processing, knowledge graph exploration, entity review, and natural language querying.

**Tech Stack:**
- **Framework:** SolidJS (fine-grained reactivity)
- **Build Tool:** Vite (fast bundler and dev server)
- **Styling:** TailwindCSS (utility-first CSS)
- **State Management:** SolidJS Stores (reactive primitives + context)
- **Graph Visualization:** Sigma.js with graphology
- **API Client:** Auto-generated from OpenAPI schema (@hey-api/typescript)
- **Testing:** Playwright (end-to-end testing)
- **Type Safety:** TypeScript 5.x

## Table of Contents

1. [Development Setup](#development-setup)
2. [Architecture Overview](#architecture-overview)
3. [API Client](#api-client)
4. [State Management](#state-management)
5. [Component Structure](#component-structure)
6. [Build & Development](#build--development)
7. [Testing](#testing)

## Development Setup

### Prerequisites

Ensure you have the following installed:
- **Bun** 1.0+ (JavaScript runtime and package manager)
- **Python 3.12+** with **uv** (for backend API)
- **PostgreSQL 14+** with pgvector extension

### Quick Start

```bash
# 1. Check system requirements
./scripts/setup_kbv2.sh check

# 2. Install dependencies and setup both backend and frontend
./scripts/setup_kbv2.sh setup

# 3. Start backend (in terminal 1)
./scripts/setup_kbv2.sh backend

# 4. Start frontend (in terminal 2)
./scripts/setup_kbv2.sh frontend

# Or start both together
./scripts/setup_kbv2.sh both
```

### Manual Setup

If you prefer manual control:

```bash
# Install Bun (if not installed)
curl -fsSL https://bun.sh/install | bash

# Navigate to frontend
cd frontend

# Install dependencies
bun install

# Generate API client (if backend is running)
bun run api:generate

# Start development server
bun run dev
```

**Access URLs:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

### Directory Structure

```
frontend/
├── src/
│   ├── api/                    # Auto-generated API client
│   │   ├── types.gen.ts        # Auto-generated TypeScript types
│   │   ├── graph-types.ts      # Manual graph-related types
│   │   ├── custom-client.ts    # Custom API client with error handling
│   │   └── index.ts            # API exports
│   ├── components/             # Reusable UI components
│   │   ├── document/          # Document viewer components
│   │   ├── graph/             # Knowledge graph visualization
│   │   ├── ingestion/         # Document ingestion monitor
│   │   └── review/            # Review queue interface
│   ├── hooks/                  # Reusable logic hooks
│   │   ├── document/          # Document-related hooks
│   │   ├── graph/             # Graph-related hooks
│   │   └── review/            # Review-related hooks
│   ├── stores/                # Global state management
│   │   ├── documentStore.ts   # Document state
│   │   ├── graphStore.ts      # Graph visualization state
│   │   ├── ingestionStore.ts  # Ingestion pipeline state
│   │   ├── reviewStore.ts     # Review queue state
│   │   └── mcpTypes.ts        # MCP protocol types
│   ├── styles/                # CSS and styling
│   │   ├── index.css          # Global styles
│   │   └── output.css         # Tailwind output (generated)
│   └── App.tsx                # Main application component
│   └── index.tsx              # Application entry point
├── tests/
│   └── e2e/                   # Playwright end-to-end tests
│       ├── phases/            # Feature-specific tests
│       └── helpers/           # Test utilities
├── test-data/                 # Test fixtures
├── openapi-schema.json        # Auto-generated API schema
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── vite.config.ts             # Vite build configuration
├── tailwind.config.js         # TailwindCSS configuration
└── openapi-ts.config.ts       # API client generation config
```

### Design Principles

1. **Reactive State Management**: SolidJS Stores for reactive shared state
2. **Type-Safe API Layer**: Auto-generated API client with TypeScript
3. **Component Composition**: Small, focused components with composable hooks
4. **User-Friendly Error Handling**: Graceful error states and user feedback
5. **Real-time Updates**: WebSocket support for live data

## API Client

### Auto-Generated Types

API types are auto-generated from the backend OpenAPI schema using `@hey-api/openapi-ts`:

```bash
# Generate from OpenAPI schema (backend must be running)
bun run api:generate

# Watch for schema changes (development mode)
bun run api:watch
```

**Generated Files:**
- `src/api/types.gen.ts` - Complete TypeScript types for all endpoints
- `src/api/index.ts` - Client exports and custom types

### Using the API Client

```typescript
import { apiClient } from './api';
import { ApiTypes } from './api';

// Ingest a document
const ingestDocument = async (filePath: string) => {
  const response = await apiClient<ApiTypes.IngestDocumentData, ApiTypes.IngestDocumentResponse>({
    method: 'POST',
    body: { file_path: filePath },
    url: '/api/v1/query/ingest',
  });
  return response;
};

// Query the knowledge graph
const query = async (naturalLanguage: string) => {
  const response = await apiClient<ApiTypes.TextToSqlData, ApiTypes.TextToSqlResponse>({
    method: 'POST',
    body: { query: naturalLanguage },
    url: '/api/v1/query/text_to_sql',
  });
  return response.data;
};
```

### Custom Client Features

The `custom-client.ts` provides enhanced error handling, request/response logging, and WebSocket support:

- **Error Interception**: Centralized error handling and user notifications
- **Request Logging**: Debug information in development mode
- **Type Safety**: Full TypeScript type inference

### Key API Endpoints

**Document Ingestion:**
- `POST /api/v1/query/ingest` - Ingest document from file path

**Knowledge Graph:**
- `GET /api/v1/query/graph` - Get knowledge graph data
- `GET /api/v1/query/graph/{id}` - Get specific graph elements

**Natural Language Query:**
- `POST /api/v1/query/text_to_sql` - Convert natural language to SQL
- `POST /api/v1/query/search_entities` - Search entities with vector similarity

**Review Queue:**
- `GET /api/v1/review/pending` - Get pending reviews
- `POST /api/v1/review/submit` - Submit review decision

## State Management

### Store Architecture

KBV2 uses SolidJS Stores for global state management with the following structure:

```
stores/
├── documentStore.ts          # Document viewing and navigation
├── graphStore.ts             # Graph visualization state
├── ingestionStore.ts         # Document ingestion pipeline
└── reviewStore.ts            # Human review queue
```

### Store Implementation Pattern

Each store follows a consistent pattern:

1. **Store State**: Reactive primitives and derived state
2. **Actions**: Functions that modify state
3. **Selectors**: Derived/computed state
4. **Initialization**: Async data fetching and setup

Example (Document Store):

```typescript
interface DocumentStore {
  documents: SchemaTypes.Document[];
  selectedDocumentId: string | null;
  loading: boolean;
  error: Error | null;
  
  // Actions
  loadDocuments: () => Promise<void>;
  selectDocument: (id: string) => void;
  
  // Selectors
  selectedDocument: () => SchemaTypes.Document | undefined;
}

// Usage in components
const [documentState, documentActions] = useContext(DocumentContext);
```

### Store Details

#### Document Store (`stores/documentStore.ts`)

Manages document lifecycle and viewing:

- **State:** Documents list, selected document, loading/error states
- **Actions:** Load documents, select document, refresh document data
- **Use cases:** Document browser, document viewer, file management

#### Graph Store (`stores/graphStore.ts`)

Handles knowledge graph visualization state:

- **State:** Graph data (nodes/edges), selected node, layout algorithm, filters
- **Actions:** Load graph data, select node, apply filters, export graph
- **Use cases:** Network visualization, entity exploration, community detection

#### Ingestion Store (`stores/ingestionStore.ts`)

Tracks document ingestion pipeline progress:

- **State:** Active ingestions, processing stages, terminal logs, errors
- **Actions:** Start ingestion, monitor progress, cancel ingestion, view logs
- **Use cases:** Real-time ingestion monitoring, pipeline visualization

#### Review Store (`stores/reviewStore.ts`)

Manages human review queue:

- **State:** Review queue items, selected item, confidence filters, pagination
- **Actions:** Load reviews, submit decision, filter by confidence, export
- **Use cases:** Entity resolution review, quality assurance, collaborative review

### Using Stores in Components

**Accessing State:**

```typescript
import { DocumentContext } from '../stores/documentStore';

function DocumentList() {
  const [documentState, documentActions] = useContext(DocumentContext);
  
  createEffect(() => {
    // Log when documents change
    console.log('Documents loaded:', documentState.documents);
  });
  
  return (
    <Show when={!documentState.loading}>
      <For each={documentState.documents}>
        {(doc) => <DocumentCard doc={doc} />}
      </For>
    </Show>
  );
}
```

**Dispatching Actions:**

```typescript
function DocumentCard(props: { doc: SchemaTypes.Document }) {
  const [, documentActions] = useContext(DocumentContext);
  
  return (
    <button onClick={() => documentActions.selectDocument(props.doc.id)}>
      {props.doc.filename}
    </button>
  );
}
```

## Component Structure

### Component Organization

```
components/
├── document/          # Document viewing and management
├── graph/            # Knowledge graph visualization
├── ingestion/        # Document processing pipeline
└── review/           # Human review interface
```

### Document Components

**DocumentViewer.tsx**
- Displays document content and metadata
- Shows entity annotations and relationships
- Supports multiple document types (PDF, text, etc.)

**DocumentBrowser.tsx**
- Lists all documents in the knowledge base
-Supports search and filtering
- Shows document status and processing history

### Graph Components

**GraphCanvas.tsx**
- Main graph visualization using Sigma.js
- Handles node/edge rendering and interactions
- Implements zoom, pan, and layout algorithms

**GraphControls.tsx**
- Graph manipulation controls (zoom, filter, layout)
- Export options (PNG, SVG, data export)
- Layout algorithm selection

**NodeTooltip.tsx**
- Contextual information on node hover
- Entity details and relationships preview
- Quick actions (expand, inspect, add to review)

### Ingestion Components

**IngestionMonitor.tsx**
- Real-time ingestion pipeline monitoring
- Shows processing stages and progress
- Terminal-style log output with search

**StageStepper.tsx**
- Visual representation of ingestion stages
- Shows completed/current/pending stages
- Includes error indicators and retry options

**TerminalLog.tsx**
- Real-time log streaming with filtering
- Search and highlight functionality
- Export log for debugging

### Review Components

**ReviewQueue.tsx**
- Lists entities needing human review
- Confidence scores and similarity metrics
- Batch operations and bulk management

**ReviewCard.tsx**
- Detailed review interface for single entity
- Side-by-side entity comparison
- Decision workflow (merge/keep/skip)

## Build & Development

### Development Commands

```bash
# Install dependencies
bun install

# Generate API types
bun run api:generate

# Start development server
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview

# Watch for API changes
bun run api:watch
```

### Code Quality

```bash
# Format code (using prettier)
npx prettier --write .

# Run TypeScript check
bun run tsc --noEmit

# Linting (if configured)
bun run lint
```

### Build Process

**Development Build:**
- Vite dev server with HMR
- Source maps enabled
- Hot module replacement
- API proxy to backend

**Production Build:**
- Minified and optimized bundles
- Tree-shaking and code splitting
- TailwindCSS purging
- Static asset optimization

### Configuration Files

**vite.config.ts**
```typescript
export default defineConfig({
  plugins: [solidPlugin(), basicSsl()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': 'ws://localhost:8000'
    }
  },
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets'
  }
});
```

**tailwind.config.js**
```javascript
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: colors.blue,
        accent: colors.purple
      }
    }
  },
  plugins: []
};
```

## Testing

### End-to-End Testing

The frontend uses Playwright for comprehensive E2E testing:

```bash
# Run all E2E tests
bun run test:e2e

# Run specific test phases
bun run test:phases

# Run in UI mode for debugging
npx playwright test --ui

# Generate test report
npx playwright show-report
```

### Test Structure

```
tests/e2e/
├── phases/
│   ├── phase1-type-bridge.spec.ts      # Document ingestion flow
│   ├── phase2-graph.test.ts            # Graph visualization
│   ├── phase3-evidence.test.ts         # Evidence search
│   ├── phase-4-control-tower.test.ts   # Review queue
│   └── phase-5-judge.test.ts           # Full flow integration
├── helpers/
│   └── backend-client.ts               # Test API client
├── test-data/                          # Test fixtures
├── global-setup.ts                     # Test environment setup
└── global-teardown.ts                  # Cleanup
```

### Writing Tests

```typescript
import { test, expect } from '@playwright/test';

test('document ingestion workflow', async ({ page }) => {
  // Navigate to app
  await page.goto('/');
  
  // Fill file path
  await page.getByLabel('File Path').fill('/path/to/test.pdf');
  
  // Click ingest button
  await page.getByRole('button', { name: 'Ingest' }).click();
  
  // Wait for success message
  await expect(page.getByText('Processing complete')).toBeVisible();
  
  // Verify document appears in list
  await expect(page.getByText('test.pdf')).toBeVisible();
});
```

### Test Data

**Location:** `frontend/tests/test-data/`

Contains sample documents and test fixtures:
- Sample PDF documents for ingestion testing
- Pre-generated entity/edge data for graph testing
- Mock review queue items for review interface testing

## Development Workflow

### Starting Development

**Terminal 1 - Backend:**
```bash
# Start backend server
uv run knowledge-base

# In separate terminal, run API type watcher
cd frontend && bun run api:watch
```

**Terminal 2 - Frontend:**
```bash
cd frontend
bun run dev
```

### Making Changes

#### API Changes

1. Modify backend API endpoints in `knowledge_base/query_api.py`
2. Restart backend server
3. Generate new API types: `bun run api:generate`
4. Update frontend components to use new types

#### Component Changes

1. Identify component to modify (check component structure)
2. Update component with new functionality
3. Add/update tests in `tests/e2e/`
4. Test manually in browser
5. Run E2E tests to verify

#### Store Changes

1. Modify store in `src/stores/[store-name].ts`
2. Add actions and state as needed
3. Update components using the store
4. Test store behavior across multiple components

### Debugging

**Frontend Console:**
- View browser console for client-side errors
- Check network tab for API request/response logs
- Use SolidJS DevTools for component inspection

**Backend Logs:**
- Check `logs/kbv2.log` for backend errors
- View API logs at http://localhost:8000/docs

**Common Issues:**

1. **API Types Not Generated:** Ensure backend is running, then `bun run api:generate`
2. **Graph Not Rendering:** Check Sigma.js dependencies and CSS
3. **Review Queue Empty:** Add test data or ingest documents with low confidence scores
4. **Build Errors:** Check TypeScript types and import paths

## Key Features

### Knowledge Graph Visualization

- Interactive Sigma.js graph with zoom/pan
- Node clustering with Leiden algorithm
- Temporal filtering and dynamic layouts
- Entity relationship exploration

### Document Processing Pipeline

- Dropzone or file path-based document ingestion
- Real-time pipeline monitoring with logs
- Multi-stage processing visualization
- Error detection and retry mechanisms

### Human Review Queue

- Priority-based review queue
- Entity similarity comparison tools
- Merge/keep/skip decision workflow
- Bulk review operations

### Natural Language Query

- Text-to-SQL translation
- Entity search with vector similarity
- Rich result visualization
- Query history and saving

## Troubleshooting

### Frontend Build Fails

- Ensure Bun is installed: `bun --version`
- Clear dependencies: `rm -rf node_modules bun.lockb && bun install`
- Check TypeScript types: `bun run tsc --noEmit`

### API Client Not Generated

- Verify backend is running on port 8000
- Check OpenAPI schema exists: `curl http://localhost:8000/openapi.json`
- Generate manually: `bun run api:generate`

### Graph Doesn't Display

- Verify data is loaded in graph store
- Check Sigma.js canvas has dimensions (height issue)
- Inspect browser console for JavaScript errors

### Tests Fail

- Ensure backend is running for E2E tests
- Check test data exists in `tests/test-data/`
- Run in UI mode: `npx playwright test --ui`

## Deployment

### Production Build

```bash
# Generate production build
cd frontend && bun run build

# Serve production build
bun run preview

# Or deploy to static hosting (Netlify, Vercel, etc.)
# Upload contents of dist/ directory
```

### Environment Variables

Create `.env` file in frontend directory:

```bash
VITE_API_URL=https://api.yourdomain.com
VITE_WEBSOCKET_URL=wss://api.yourdomain.com
VITE_ENV=production
```

## Best Practices

1. **Component Design**: Keep components small and focused on single responsibilities
2. **State Management**: Use stores for shared state, local state for component-specific data
3. **Type Safety**: Always type props and state with generated API types
4. **Error Handling**: Use the custom API client for consistent error handling
5. **Testing**: Write Playwright tests for user workflows
6. **Performance**: Use SolidJS fine-grained reactivity to avoid unnecessary renders
7. **Graph UX**: Optimize graph performance for large datasets (>1000 nodes)
8. **Accessibility**: Use semantic HTML and ARIA attributes

## Related Documentation

- [Backend API Guide](../docs/api/endpoints.md)
- [System Architecture](../docs/architecture/system_overview.md)
- [Testing Strategy](../tests/e2e/phases/readme.md)
- [Project Overview](../PROJECT_GUIDE.md)

## Getting Help

- **API Issues**: Check backend logs and API docs at `/docs`
- **Graph Issues**: Inspect Sigma.js canvas and data structure
- **Store Issues**: Use SolidJS DevTools to inspect store state
- **Build Issues**: Check TypeScript types and dependencies

---

**Last Updated:** January 2026  
**Frontend Version:** 0.1.0  
**Compatible Backend:** KBV2 v1.0.0+