import { For, Show, createSignal, onMount, createEffect } from 'solid-js';
import type { IngestionStoreType } from '../../stores/ingestionStore';

interface TerminalLogProps {
  ingestionStore: IngestionStoreType['ingestionStore'];
}

const TerminalLog = (props: TerminalLogProps) => {
  const { ingestionStore } = props;
  let logContainerRef: HTMLDivElement | undefined;
  const [scrollTarget, setScrollTarget] = createSignal<HTMLElement | null>(null);

  createEffect(() => {
    if (logContainerRef && ingestionStore.logs && ingestionStore.logs.length > 0) {
      logContainerRef.scrollTop = logContainerRef.scrollHeight;
    }
  });

  const getLogColor = (level: string) => {
    switch (level) {
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      case 'debug': return 'text-gray-500';
      default: return 'text-gray-300';
    }
  };

  return (
    <div class="flex-1 flex flex-col min-h-0 TerminalLog">
      <div class="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700">
        <h3 class="text-sm font-semibold text-gray-300">Ingestion Log</h3>
        <div class="flex items-center space-x-2">
          <span class={`text-xs px-2 py-1 rounded-full ${
            ingestionStore.connectionStatus === 'connected'
              ? 'bg-green-900 text-green-300'
              : ingestionStore.connectionStatus === 'connecting'
              ? 'bg-yellow-900 text-yellow-300'
              : 'bg-red-900 text-red-300'
          }`}>
            {ingestionStore.connectionStatus}
          </span>
          <button
            onClick={() => {
              // Clear logs would be implemented here
            }}
            class="text-xs text-gray-400 hover:text-gray-200 px-2 py-1 rounded hover:bg-gray-700"
          >
            Clear
          </button>
        </div>
      </div>

      <div
        ref={logContainerRef}
        class="flex-1 bg-black p-3 font-mono text-xs overflow-y-auto"
      >
        <For each={ingestionStore.logs}>
          {(log) => (
            <div class={`mb-1 ${getLogColor(log.level)}`}>
              <span class="text-gray-600 mr-2">
                [{new Date(log.timestamp).toLocaleTimeString()}]
              </span>
              <span class="uppercase mr-2">{log.level}</span>
              <span>{log.message}</span>
            </div>
          )}
        </For>

        <Show when={!ingestionStore.logs || ingestionStore.logs.length === 0}>
          <div class="text-gray-500 italic">No logs yet. Start an ingestion to see logs here.</div>
        </Show>
      </div>
    </div>
  );
};

export default TerminalLog;