import { createStore } from 'solid-js/store';

type GraphSummary = any;
type GraphNode = any;
type GraphEdge = any;

export interface GraphStore {
  currentGraphId: string | null;
  graphData: GraphSummary | null;
  activeNode: GraphNode | null;
  loading: boolean;
  error: string | null;
  expandedNodes: Set<string>;
}

const [graphStore, setGraphStore] = createStore<GraphStore>({
  currentGraphId: null,
  graphData: null,
  activeNode: null,
  loading: false,
  error: null,
  expandedNodes: new Set(),
});

export const createGraphStore = () => {
  const setGraph = (graphId: string, data: GraphSummary) => {
    setGraphStore({
      currentGraphId: graphId,
      graphData: data,
      loading: false,
      error: null,
    });
  };

  const setActiveNode = (node: GraphNode | null) => {
    setGraphStore('activeNode', node);
  };

  const expandNode = (nodeId: string) => {
    setGraphStore('expandedNodes', (prev) => new Set(prev).add(nodeId));
  };

  const clearError = () => {
    setGraphStore('error', null);
  };

  const setLoading = (loading: boolean) => {
    setGraphStore('loading', loading);
  };

  const setError = (error: string) => {
    setGraphStore({
      error,
      loading: false,
    });
  };

  return {
    graphStore,
    setGraph,
    setActiveNode,
    expandNode,
    clearError,
    setLoading,
    setError,
  };
};

export type GraphStoreType = ReturnType<typeof createGraphStore>;
