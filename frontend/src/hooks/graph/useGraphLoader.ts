import { createResource, createSignal, onMount, onCleanup, createEffect, createComputed } from 'solid-js';
import { getGraphSummaryApiV1GraphsGraphIdSummaryGet, getNeighborhoodApiV1GraphsGraphIdNodesNodeIdNeighborhoodGet } from '../../api/sdk.gen';
import type { GraphStoreType } from '../../stores/graphStore';

interface UseGraphLoaderProps {
  graphStore: GraphStoreType;
}

interface GraphLoader {
  loadGraph: (graphId: string, level?: number) => Promise<void>;
  loadNeighborhood: (nodeId: string, depth?: number) => Promise<any>;
  loading: () => boolean;
}

export function useGraphLoader({ graphStore }: UseGraphLoaderProps): GraphLoader {
  const [loading, setLoading] = createSignal(false);

  const loadGraph = async (graphId: string, level: number = 0) => {
    setLoading(true);
    graphStore.setLoading(true);
    graphStore.clearError();

    try {
      const response = await getGraphSummaryApiV1GraphsGraphIdSummaryGet({
        params: {
          path: { graph_id: graphId },
          query: {
            level,
            min_community_size: 3,
            include_metrics: true,
          },
        },
      });

      if (!response.data) {
        throw new Error('No response from server');
      }

      graphStore.setGraph(graphId, response.data);
      setLoading(false);
      graphStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load graph:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      graphStore.setError(`Failed to load graph: ${errorMessage}`);
      setLoading(false);
    }
  };

  const loadNeighborhood = async (nodeId: string, depth: number = 1) => {
    const graphId = graphStore.graphStore.currentGraphId;
    if (!graphId) {
      throw new Error('No graph loaded');
    }

    setLoading(true);
    graphStore.setLoading(true);

    try {
      const response = await getNeighborhoodApiV1GraphsGraphIdNodesNodeIdNeighborhoodGet({
        params: {
          path: {
            graph_id: graphId,
            node_id: nodeId,
          },
          query: {
            depth,
            direction: 'bidirectional',
            min_confidence: 0.7,
            max_nodes: 1000,
          },
        }
      });

      if (!response.data) {
        throw new Error('No response from server');
      }

      setLoading(false);
      graphStore.setLoading(false);
      graphStore.expandNode(nodeId);

      return response.data;
    } catch (error) {
      console.error('Failed to load neighborhood:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      graphStore.setError(`Failed to load neighborhood: ${errorMessage}`);
      setLoading(false);
      throw error;
    }
  };

  return {
    loadGraph,
    loadNeighborhood,
    loading,
  };
}
