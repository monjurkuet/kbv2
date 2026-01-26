import { createSignal, Show, Match, Switch, onMount } from 'solid-js';
import StageStepper from './StageStepper';
import TerminalLog from './TerminalLog';
import type { IngestionStoreType } from '../../stores/ingestionStore';

interface IngestionMonitorProps {
  ingestionStore: IngestionStoreType['ingestionStore'];
}

const IngestionMonitor = (props: IngestionMonitorProps) => {
  const { ingestionStore } = props;
  const [filePath, setFilePath] = createSignal('');
  const [documentName, setDocumentName] = createSignal('');
  const [domain, setDomain] = createSignal('general');
  const [showConfig, setShowConfig] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  onMount(() => {
    ingestionStore.connect();
  });

  const handleStartIngestion = () => {
    try {
      setError(null);
      if (filePath().trim()) {
        ingestionStore.startIngestion(filePath().trim(), documentName().trim() || undefined, domain() || undefined);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start ingestion');
    }
  };

  const handleReset = () => {
    try {
      setError(null);
      ingestionStore.reset();
      setFilePath('');
      setDocumentName('');
      setDomain('general');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reset');
    }
  };

  const getConnectionStatusColor = () => {
    switch (ingestionStore.connectionStatus) {
      case 'connected': return 'bg-green-100 text-green-800 border-green-200';
      case 'connecting': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'error': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div class="w-full h-full flex flex-col">
      {/* Header */}
      <div class="bg-white border-b border-gray-200 px-6 py-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-2xl font-bold text-gray-900">Ingestion Control Tower</h2>
            <p class="text-sm text-gray-600">Monitor document processing pipeline in real-time</p>
          </div>

          <div class="flex items-center space-x-3">
            <div class={`px-3 py-1 rounded-full text-xs font-semibold border ${getConnectionStatusColor()}`}>
              {(ingestionStore.connectionStatus || 'disconnected').toUpperCase()}
            </div>

            <button
              onClick={() => setShowConfig(!showConfig())}
              class="px-3 py-1 text-xs font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            >
              Configure
            </button>
          </div>
        </div>
      </div>

      {/* Error Display */}
      <Show when={error()}>
        <div class="bg-red-50 border-b border-red-200 px-6 py-3">
          <p class="text-sm text-red-800">{error()}</p>
        </div>
      </Show>

      {/* Configuration Panel */}
      <Show when={showConfig()}>
        <div class="bg-blue-50 border-b border-blue-200 px-6 py-4">
          <div class="flex items-end space-x-4">
            <div class="flex-1">
              <label class="block text-xs font-semibold text-blue-900 mb-1">
                File Path
              </label>
              <input
                type="text"
                value={filePath()}
                onInput={(e) => setFilePath(e.currentTarget.value)}
                placeholder="/path/to/document.pdf"
                class="w-full px-3 py-2 text-sm border border-blue-300 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div class="flex-1">
              <label class="block text-xs font-semibold text-blue-900 mb-1">
                Document Name (Optional)
              </label>
              <input
                type="text"
                value={documentName()}
                onInput={(e) => setDocumentName(e.currentTarget.value)}
                placeholder="My Document"
                class="w-full px-3 py-2 text-sm border border-blue-300 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div class="w-48">
              <label class="block text-xs font-semibold text-blue-900 mb-1">
                Domain
              </label>
              <select
                value={domain()}
                onInput={(e) => setDomain(e.currentTarget.value)}
                class="w-full px-3 py-2 text-sm border border-blue-300 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="general">General</option>
                <option value="cybersecurity">Cybersecurity</option>
                <option value="finance">Finance</option>
                <option value="healthcare">Healthcare</option>
                <option value="legal">Legal</option>
              </select>
            </div>

            <div class="flex space-x-2">
              <button
                onClick={handleStartIngestion}
                disabled={!filePath().trim() || ingestionStore.isRunning}
                class="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed rounded-md transition-colors"
              >
                {ingestionStore.isRunning ? 'Running...' : 'Start Ingestion'}
              </button>

              <button
                onClick={handleReset}
                disabled={ingestionStore.isRunning}
                class="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed disabled:text-gray-400 rounded-md transition-colors"
              >
                Reset
              </button>
            </div>
          </div>
        </div>
      </Show>

      {/* Main Content */}
      <div class="flex-1 flex min-h-0">
        {/* Stage Stepper (Left Panel) */}
        <div class="w-96 bg-gray-50 border-r border-gray-200 p-6 overflow-y-auto">
          <StageStepper ingestionStore={ingestionStore} />
        </div>

        {/* Terminal Log (Right Panel) */}
        <div class="flex-1 flex flex-col">
          <TerminalLog ingestionStore={ingestionStore} />
        </div>
      </div>
    </div>
  );
};

export default IngestionMonitor;