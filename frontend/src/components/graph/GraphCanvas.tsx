import { createEffect, createSignal, onMount, onCleanup, Show, createMemo, createResource, ErrorBoundary } from 'solid-js';
import Sigma from 'sigma';
import type { Sigma as SigmaType } from 'sigma';
import Graph from 'graphology';
import { EDGE_COLORS, ENTITY_COLORS, ENTITY_TYPES, NODE_SIZES, SIGMA_SETTINGS } from './constants';
import { useGraphLoader } from '../../hooks/graph/useGraphLoader';
import type { GraphStoreType } from '../../stores/graphStore';
import type { GraphNode as ApiGraphNode, GraphEdge as ApiGraphEdge } from '../../api';
import GraphControls from './GraphControls';
import NodeTooltip from './NodeTooltip';

interface GraphCanvasProps {
  graphStore: GraphStoreType;
  graphId: string;
}

const GraphCanvas = (props: GraphCanvasProps) => {
  const [error, setError] = createSignal<string | null>(null);
  let containerRef: HTMLDivElement | undefined;
  let sigmaInstance: SigmaType | undefined;
  const [hoveredNode, setHoveredNode] = createSignal<string | null>(null);
  const [tooltipPos, setTooltipPos] = createSignal<{ x: number; y: number } | null>(null);

  let loadGraph: ((graphId: string, level?: number) => Promise<void>) | undefined;
  let loadNeighborhood: ((nodeId: string, depth?: number) => Promise<{
    center_node: ApiGraphNode;
    nodes: ApiGraphNode[];
    edges: ApiGraphEdge[];
  }>) | undefined;
  let loading: () => boolean;

  try {
    const loader = useGraphLoader({ graphStore: props.graphStore });
    loadGraph = loader.loadGraph;
    loadNeighborhood = loader.loadNeighborhood;
    loading = loader.loading;
  } catch (e) {
    console.error('Failed to initialize graph loader:', e);
    setError('Failed to initialize graph loader');
    loading = () => false;
  }

  const graph = createMemo(() => {
    const storeData = props.graphStore.graphStore.graphData;
    if (!storeData) return null;

    const graphInstance = new Graph({ type: 'directed', multi: true });

    storeData.nodes.forEach((node) => {
      const color = node.attributes.community_id ? ENTITY_COLORS.community : ENTITY_COLORS.center;
      const size = node.attributes.size || Math.max(NODE_SIZES.min, Math.log(node.attributes.entity_count || 1) * 3);

      graphInstance.addNode(node.key, {
        x: node.attributes.x || Math.random() * 100,
        y: node.attributes.y || Math.random() * 100,
        size: size,
        color: color,
        label: node.label,
        ...node.attributes,
      });
    });

    storeData.edges.forEach((edge) => {
      const alpha = edge.attributes.confidence ? edge.attributes.confidence * 0.3 + 0.1 : 0.3;
      graphInstance.addEdge(edge.source, edge.target, {
        color: `rgba(0, 0, 0, ${alpha})`,
        size: edge.attributes.weight || 1,
        ...edge.attributes,
      });
    });

    return graphInstance;
  });

  onMount(async () => {
    if (!containerRef) return;

    try {
      if (loadGraph) {
        await loadGraph(props.graphId);
      }

      const currentGraph = graph();
      if (!currentGraph) return;

    sigmaInstance = new Sigma(currentGraph, containerRef, {
      renderEdgeLabels: true,
      enableEdgeHoverEvents: true,
      enableNodeHoverEvents: true,
      labelDensity: 0.07,
      labelGridCellSize: 100,
      labelRenderedSizeThreshold: 10,
      labelFont: 'Arial, sans-serif',
      labelSize: 12,
    });

    sigmaInstance.on('clickNode', async (event) => {
      const nodeId = event.node;
      const nodeData = currentGraph.getNodeAttributes(nodeId) as any;

      if (!props.graphStore.graphStore.expandedNodes.has(nodeId)) {
        props.graphStore.setActiveNode(nodeData);

        try {
          if (loadNeighborhood) {
            const neighborhood = await loadNeighborhood(nodeId);

            if (neighborhood.nodes && neighborhood.edges) {
              neighborhood.nodes.forEach((node: ApiGraphNode) => {
                if (!currentGraph.hasNode(node.key)) {
                  const isNeighbor = node.attributes.community_id !== nodeData.community_id;
                  const color = isNeighbor ? ENTITY_COLORS.neighbor : ENTITY_COLORS.center;
                  const size = node.attributes.size || NODE_SIZES.min;

                  currentGraph.addNode(node.key, {
                    x: node.attributes.x || (Math.random() - 0.5) * 200 + (nodeData.x || 0),
                    y: node.attributes.y || (Math.random() - 0.5) * 200 + (nodeData.y || 0),
                    size: size,
                    color: color,
                    label: node.label,
                    ...node.attributes,
                  });
                }
              });

              neighborhood.edges.forEach((edge: ApiGraphEdge) => {
                if (!currentGraph.hasEdge(edge.source, edge.target)) {
                  const alpha = edge.attributes.confidence ? edge.attributes.confidence * 0.3 + 0.1 : 0.3;
                  currentGraph.addEdge(edge.source, edge.target, {
                    color: `rgba(0, 0, 0, ${alpha})`,
                    size: edge.attributes.weight || 1,
                    ...edge.attributes,
                  });
                }
              });

              sigmaInstance?.refresh();
            }
          }
        } catch (error) {
          console.error('Failed to load neighborhood:', error);
        }
      }
    });

    sigmaInstance.on('enterNode', (event) => {
      setHoveredNode(event.node);
      const pos = sigmaInstance?.viewportToFramedGraph(event);
      if (pos) {
        const canvasPos = sigmaInstance?.graphToCanvas(pos);
        if (canvasPos) setTooltipPos({ x: canvasPos.x + 10, y: canvasPos.y - 30 });
      }
    });

    sigmaInstance.on('leaveNode', () => {
      setHoveredNode(null);
      setTooltipPos(null);
    });
    } catch (e) {
      console.error('Failed to initialize graph:', e);
      setError('Failed to initialize graph');
    }
  });

  createEffect(() => {
    const currentGraph = graph();
    if (sigmaInstance && currentGraph) {
      sigmaInstance.setGraph(currentGraph);
      sigmaInstance.refresh();
    }
  });

  onCleanup(() => {
    if (sigmaInstance) {
      sigmaInstance.kill();
    }
  });

  return (
    <div class="relative w-full h-full bg-gray-50">
      <Show when={error()}>
        <div class="absolute inset-0 flex items-center justify-center bg-red-50">
          <div class="bg-white rounded-lg shadow-lg p-6 max-w-md">
            <h3 class="text-red-800 font-semibold mb-2">Error</h3>
            <p class="text-red-600 text-sm">{error()}</p>
          </div>
        </div>
      </Show>

      <div ref={containerRef} class="w-full h-full" />

      <Show when={hoveredNode() && tooltipPos()}>
        <NodeTooltip nodeId={hoveredNode()!} position={tooltipPos()!} />
      </Show>

      <GraphControls sigma={sigmaInstance} />

      <Show when={loading()}>
        <div class="absolute top-4 right-4 bg-white rounded-lg shadow-lg px-4 py-2">
          <div class="flex items-center space-x-2">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600" />
            <span class="text-sm text-gray-700">Loading graph...</span>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default GraphCanvas;
