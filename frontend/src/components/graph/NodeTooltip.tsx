import { Show, createMemo } from 'solid-js';
import type { GraphNode } from '../../api';

interface NodeTooltipProps {
  nodeId: string;
  position: { x: number; y: number };
}

const NodeTooltip = (props: NodeTooltipProps) => {
  const nodeData = createMemo(() => {
    return {
      label: props.nodeId,
      type: 'entity',
      confidence: 0.95,
      entity_count: 5,
    };
  });

  return (
    <div
      class="absolute z-50 bg-gray-900 text-white text-xs rounded-lg shadow-lg p-3 pointer-events-none"
      style={{
        left: `${props.position.x}px`,
        top: `${props.position.y}px`,
        transform: 'translate(-50%, -100%)',
      }}
    >
      <div class="font-semibold text-sm mb-1">{nodeData().label}</div>
      
      <div class="space-y-0.5">
        <div class="flex justify-between">
          <span class="text-gray-400">Type:</span>
          <span class="ml-2">{nodeData().type}</span>
        </div>
        
        <Show when={nodeData().confidence !== undefined}>
          <div class="flex justify-between">
            <span class="text-gray-400">Confidence:</span>
            <span class="ml-2">{Math.round((nodeData().confidence || 0) * 100)}%</span>
          </div>
        </Show>
        
        <Show when={nodeData().entity_count !== undefined}>
          <div class="flex justify-between">
            <span class="text-gray-400">Entities:</span>
            <span class="ml-2">{nodeData().entity_count}</span>
          </div>
        </Show>
      </div>
      
      <div class="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-full">
        <div class="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
      </div>
    </div>
  );
};

export default NodeTooltip;