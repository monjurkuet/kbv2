import { createResource, createSignal, onMount, onCleanup, createEffect, createComputed } from 'solid-js';
import { apiClient } from '../../api/custom-client';
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
      const response = await apiClient.GET('/api/v1/graphs/{graph_id}:summary', {
        params: {
          path: { graph_id: graphId },
          query: {
            level,
            min_community_size: 3,
            include_metrics: true,
          },
        },
      });

      if (!response) {
        throw new Error('No response from server');
      }

      graphStore.setGraph(graphId, response);
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
      const response = await apiClient.GET(
        '/api/v1/graphs/{graph_id}/nodes/{node_id}:neighborhood',
        {
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
          },
        }
      );

      if (!response) {
        throw new Error('No response from server');
      }

      setLoading(false);
      graphStore.setLoading(false);
      graphStore.expandNode(nodeId);

      return response;
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
