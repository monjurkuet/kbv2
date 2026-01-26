import { Router, Route, A, useParams } from '@solidjs/router';
import ReviewQueue from './components/review/ReviewQueue';
import ReviewCard from './components/review/ReviewCard';
import GraphCanvas from './components/graph/GraphCanvas';
import DocumentViewer from './components/document/DocumentViewer';
import IngestionMonitor from './components/ingestion/IngestionMonitor';
import { createReviewStore } from './stores/reviewStore';
import { createIngestionStore } from './stores/ingestionStore';
import { createDocumentStore } from './stores/documentStore';
import { createGraphStore } from './stores/graphStore';
import './styles/index.css';

const AppLayout = (props: { children: any, title: string, subtitle: string }) => (
  <div class="w-screen h-screen flex flex-col bg-gray-50">
    {/* Header */}
    <header class="bg-gray-900 text-white p-4 shadow-md">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-2xl font-bold">{props.title}</h1>
          <p class="text-sm text-gray-300">{props.subtitle}</p>
        </div>
        
        <div class="flex items-center space-x-4">
          <div class="flex items-center space-x-2">
            <A
              href="/queue"
              class="px-4 py-2 text-sm font-semibold rounded-md transition-colors bg-gray-800 text-gray-300 hover:bg-gray-700"
              activeClass="bg-white text-gray-900"
            >
              Queue
            </A>
            <A
              href="/control-tower"
              class="px-4 py-2 text-sm font-semibold rounded-md transition-colors bg-gray-800 text-gray-300 hover:bg-gray-700"
              activeClass="bg-white text-gray-900"
            >
              Ingestion
            </A>
            <A
              href="/review"
              class="px-4 py-2 text-sm font-semibold rounded-md transition-colors bg-gray-800 text-gray-300 hover:bg-gray-700"
              activeClass="bg-white text-gray-900"
            >
              Review
            </A>
          </div>
        </div>
      </div>
    </header>

    {/* Navigation */}
    <nav class="bg-white border-b border-gray-200 px-4 py-2">
      <div class="flex space-x-4 text-sm">
        <A href="/queue" class="text-blue-600 hover:text-blue-800 font-medium" activeClass="text-blue-800">
          Review Queue
        </A>
        <A href="/ingestion" class="text-gray-600 hover:text-gray-800" activeClass="text-gray-800">
          Ingestion Monitor
        </A>
        <A href="/graph" class="text-gray-600 hover:text-gray-800" activeClass="text-gray-800">
          Knowledge Graph
        </A>
      </div>
    </nav>

    {/* Main Content */}
    <main class="flex-1 overflow-hidden">
      {props.children}
    </main>

    {/* Footer */}
    <footer class="bg-gray-100 border-t border-gray-200 px-4 py-2 text-xs text-gray-500 text-center">
      <div class="flex justify-between items-center">
        <span>KBV2 Knowledge Engine v1.0.0</span>
        <span>Backend: http://localhost:8765</span>
      </div>
    </footer>
  </div>
);

const DocumentRoute = () => {
  const params = useParams();
  const documentId = params.documentId || "00000000-0000-0000-0000-000000000000";
  const documentStore = createDocumentStore();
  return (
    <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 3: Evidence Locker">
      <div data-testid="document-container">
        <DocumentViewer documentStore={documentStore} documentId={documentId} />
      </div>
    </AppLayout>
  );
};

const GraphRoute = () => {
  const params = useParams();
  const graphId = params.graphId || "00000000-0000-0000-0000-000000000000";
  const graphStore = createGraphStore();
  return (
    <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 2: Knowledge Explorer">
      <div data-testid="graph-container" class="flex-1 w-full h-full">
        <GraphCanvas graphStore={graphStore} graphId={graphId} />
      </div>
    </AppLayout>
  );
};

const ReviewDetailRoute = () => {
  const params = useParams();
  const reviewStore = createReviewStore();
  return (
    <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 5: Judge - Review Details">
      <ReviewCard reviewStore={reviewStore} reviewId={params.reviewId} />
    </AppLayout>
  );
};

const App = () => {
  return (
    <Router>
      <Route path="/" component={() => {
        const reviewStore = createReviewStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 5: Judge - Review Queue">
            <div data-testid="review-container">
              <ReviewQueue reviewStore={reviewStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/queue" component={() => {
        const reviewStore = createReviewStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 5: Judge - Review Queue">
            <div data-testid="review-container">
              <ReviewQueue reviewStore={reviewStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/document/:documentId" component={DocumentRoute} />
      <Route path="/graph" component={() => {
        const graphStore = createGraphStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 2: Knowledge Explorer">
            <div data-testid="graph-container" class="flex-1 w-full h-full">
              <GraphCanvas graphStore={graphStore} graphId="00000000-0000-0000-0000-000000000000" />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/graph/:graphId" component={GraphRoute} />
      <Route path="/control-tower" component={() => {
        const { ingestionStore } = createIngestionStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 4: Control Tower">
            <div data-testid="control-tower-container">
              <IngestionMonitor ingestionStore={ingestionStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/ingestion" component={() => {
        const { ingestionStore } = createIngestionStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 4: Control Tower">
            <div data-testid="control-tower-container">
              <IngestionMonitor ingestionStore={ingestionStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/review" component={() => {
        const reviewStore = createReviewStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 5: Judge - Review Queue">
            <div data-testid="review-container">
              <ReviewQueue reviewStore={reviewStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/review-queue" component={() => {
        const reviewStore = createReviewStore();
        return (
          <AppLayout title="KBV2 Knowledge Explorer" subtitle="Phase 5: Judge - Review Queue">
            <div data-testid="review-container">
              <ReviewQueue reviewStore={reviewStore} />
            </div>
          </AppLayout>
        );
      }} />
      <Route path="/review/:reviewId" component={ReviewDetailRoute} />
    </Router>
  );
};

export default App;