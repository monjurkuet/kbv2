import { For, Show } from 'solid-js';
import type { IngestionStoreType, Stage } from '../../stores/ingestionStore';

interface StageStepperProps {
  ingestionStore: IngestionStoreType['ingestionStore'];
}

const StageStepper = (props: StageStepperProps) => {
  const { ingestionStore } = props;

  const getStageIcon = (status: Stage['status']) => {
    switch (status) {
      case 'completed': return '✓';
      case 'running': return '⟳';
      case 'failed': return '✗';
      default: return '○';
    }
  };

  const getStageColor = (status: Stage['status']) => {
    switch (status) {
      case 'completed': return 'text-green-600';
      case 'running': return 'text-blue-600 animate-pulse';
      case 'failed': return 'text-red-600';
      default: return 'text-gray-400';
    }
  };

  const getStatusBadge = (status: Stage['status']) => {
    switch (status) {
      case 'completed': return { text: 'Complete', color: 'bg-green-100 text-green-800' };
      case 'running': return { text: 'Processing', color: 'bg-blue-100 text-blue-800' };
      case 'failed': return { text: 'Failed', color: 'bg-red-100 text-red-800' };
      default: return { text: 'Pending', color: 'bg-gray-100 text-gray-800' };
    }
  };

  return (
    <div class="w-full StageStepper">
      <div class="mb-4">
        <div class="flex justify-between items-center mb-2">
          <span class="text-sm font-medium text-gray-700">
            Stage {ingestionStore.currentStage || 0} of {ingestionStore.stages?.length || 0}
          </span>
          <span class="text-sm text-gray-500">
            {Math.round(ingestionStore.overallProgress || 0)}% Complete
          </span>
        </div>
        <div class="w-full bg-gray-200 rounded-full h-2">
          <div
            class="bg-blue-500 h-2 rounded-full transition-all duration-300"
            style={{ width: `${ingestionStore.overallProgress}%` }}
          />
        </div>
      </div>

      <div class="space-y-3">
        <For each={ingestionStore.stages || []}>
          {(stage) => {
            const badge = getStatusBadge(stage.status);
            return (
              <div class={`flex items-start space-x-3 p-3 rounded-lg border ${
                stage.status === 'running'
                  ? 'bg-blue-50 border-blue-200'
                  : stage.status === 'completed'
                  ? 'bg-green-50 border-green-200'
                  : stage.status === 'failed'
                  ? 'bg-red-50 border-red-200'
                  : 'bg-gray-50 border-gray-200'
              }`}>
                <div class="flex-shrink-0 mt-0.5">
                  <span class={`text-lg font-bold ${getStageColor(stage.status)}`}>
                    {getStageIcon(stage.status)}
                  </span>
                </div>

                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-2">
                      <span class="text-sm font-semibold text-gray-900">
                        {stage.number}. {stage.name}
                      </span>
                      <span class={`text-xs px-2 py-0.5 rounded-full ${badge.color}`}>
                        {badge.text}
                      </span>
                    </div>
                    <Show when={stage.durationMs}>
                      <span class="text-xs text-gray-500 font-mono">
                        {stage.durationMs}ms
                      </span>
                    </Show>
                  </div>

                  <Show when={stage.message}>
                    <div class="mt-1 text-xs text-gray-600">
                      {stage.message}
                    </div>
                  </Show>
                </div>
              </div>
            );
          }}
        </For>
      </div>

      <Show when={ingestionStore.totalDurationMs > 0}>
        <div class="mt-4 p-3 bg-gray-100 rounded-lg">
          <div class="flex items-center justify-between">
            <span class="text-sm font-medium text-gray-700">Total Duration</span>
            <span class="text-sm font-mono text-gray-900">
              {ingestionStore.totalDurationMs}ms
            </span>
          </div>
          <Show when={ingestionStore.documentId}>
            <div class="mt-1 text-xs text-gray-600 font-mono">
              Document: {ingestionStore.documentId?.slice(0, 8)}...
            </div>
          </Show>
        </div>
      </Show>
    </div>
  );
};

export default StageStepper;