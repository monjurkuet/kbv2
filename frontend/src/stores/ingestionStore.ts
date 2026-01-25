import { createStore } from 'solid-js/store';

export interface Stage {
  number: number;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime?: number;
  durationMs?: number;
  message?: string;
}

export interface LogEntry {
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  timestamp: string;
}

export interface IngestionState {
  documentId: string | null;
  stages: Stage[];
  logs: LogEntry[];
  overallProgress: number;
  isRunning: boolean;
  connectionStatus: 'disconnected' | 'connecting' | 'connected' | 'error';
  ws: WebSocket | null;
  currentStage: number;
  totalDurationMs: number;
}

export const STAGES = [
  { number: 1, name: 'Create Document' },
  { number: 2, name: 'Partition Document' },
  { number: 3, name: 'Extract Knowledge' },
  { number: 4, name: 'Embed Content' },
  { number: 5, name: 'Resolve Entities' },
  { number: 6, name: 'Cluster Entities' },
  { number: 7, name: 'Generate Reports' },
  { number: 8, name: 'Update Domain' },
  { number: 9, name: 'Complete' }
] as const;

const [ingestionStore, setIngestionStore] = createStore<IngestionState>({
  documentId: null,
  stages: STAGES.map(s => ({ ...s, status: 'pending' })),
  logs: [],
  overallProgress: 0,
  isRunning: false,
  connectionStatus: 'disconnected',
  ws: null,
  currentStage: 0,
  totalDurationMs: 0,
});

export const createIngestionStore = () => {
  const [store, setStore] = createStore<IngestionState>({
    documentId: null,
    stages: STAGES.map(s => ({ ...s, status: 'pending' })),
    logs: [],
    overallProgress: 0,
    isRunning: false,
    connectionStatus: 'disconnected',
    ws: null,
    currentStage: 0,
    totalDurationMs: 0,
  });

  const addLog = (level: LogEntry['level'], message: string) => {
    setStore('logs', logs => [...logs, {
      level,
      message,
      timestamp: new Date().toISOString()
    }]);
  };

  const connect = () => {
    console.log('Connecting to MCP WebSocket...');
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setStore({
        ws,
        connectionStatus: 'connected'
      });
      addLog('info', 'Connected to KBV2 MCP server');
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
          updateStage(data.stage, data.status, data.durationMs, data.message);
          
          const completedStages = store.stages.filter(s => s.status === 'completed').length;
          const progress = (completedStages / 9) * 100;
          setStore('overallProgress', progress);
          
        } else if (data.type === 'log') {
          addLog(data.level, data.message);
          
        } else if (data.type === 'complete') {
          setStore({
            isRunning: false,
            totalDurationMs: data.totalDurationMs || 0
          });
          addLog(
            data.status === 'completed' ? 'info' : 'error',
            `Ingestion ${data.status} in ${data.totalDurationMs || 0}ms`
          );
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStore('connectionStatus', 'error');
      addLog('error', `WebSocket error: ${error}`);
    };
    
    ws.onclose = () => {
      console.warn('WebSocket disconnected');
      setStore({
        connectionStatus: 'disconnected',
        ws: null
      });
      addLog('warning', 'Disconnected from server');
      
      if (store.isRunning) {
        setTimeout(connect, 5000);
      }
    };
  };
  
  const disconnect = () => {
    if (store.ws) {
      store.ws.close();
    }
  };
  
  const updateStage = (
    stageNumber: number,
    status: Stage['status'],
    durationMs?: number,
    message?: string
  ) => {
    setStore('stages', s => s.number === stageNumber, {
      status: status === 'started' ? 'running' : status === 'completed' ? 'completed' : status === 'failed' ? 'failed' : 'pending',
      startTime: status === 'started' ? Date.now() : undefined,
      durationMs,
      message
    });
    
    if (status === 'started') {
      setStore('currentStage', stageNumber);
    }
  };
  
  const startIngestion = async (filePath: string, documentName?: string, domain?: string) => {
    connect();
    
    setStore({
      isRunning: true,
      stages: STAGES.map(s => ({ ...s, status: 'pending' })),
      logs: [],
      overallProgress: 0,
      currentStage: 0,
      documentId: `doc_${Date.now()}`
    });
    
    if (store.ws?.readyState === WebSocket.OPEN) {
      const request = {
        method: 'kbv2/ingest_document',
        params: {
          file_path: filePath,
          document_name: documentName,
          domain: domain
        },
        id: `req_${Date.now()}`
      };
      
      store.ws.send(JSON.stringify(request));
      addLog('info', `Starting ingestion: ${filePath}`);
    } else {
      addLog('error', 'WebSocket not connected. Cannot start ingestion.');
    }
  };
  
  const reset = () => {
    setStore({
      documentId: null,
      stages: STAGES.map(s => ({ ...s, status: 'pending' })),
      logs: [],
      overallProgress: 0,
      isRunning: false,
      currentStage: 0,
      totalDurationMs: 0
    });
  };
  
  const clearLogs = () => {
    setStore('logs', []);
  };

  const ingestionStore = {
    ...store,
    connect,
    disconnect,
    startIngestion,
    reset,
    clearLogs,
    addLog
  };
  
  return {
    ingestionStore
  };
};

export type IngestionStoreType = ReturnType<typeof createIngestionStore>;