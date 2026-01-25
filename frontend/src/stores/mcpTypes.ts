export interface MCPMessage {
  method: string;
  params: Record<string, any>;
  id?: string;
}

export interface MCPResponse {
  result?: any;
  error?: string;
  id?: string;
}

export interface IngestionProgress {
  type: 'progress';
  documentId: string;
  stage: number;
  stageName: string;
  status: 'started' | 'in_progress' | 'completed' | 'failed';
  message: string;
  durationMs?: number;
  timestamp: string;
}

export interface LogMessage {
  type: 'log';
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
  timestamp: string;
}

export interface IngestionComplete {
  type: 'complete';
  documentId: string;
  status: 'completed' | 'failed';
  totalDurationMs: number;
  timestamp: string;
}

export interface WebSocketConfig {
  url: string;
  reconnectDelay: number;
  maxReconnectDelay: number;
  requestTimeout: number;
}

export const DEFAULT_WS_CONFIG: WebSocketConfig = {
  url: 'ws://localhost:8000/ws',
  reconnectDelay: 1000,
  maxReconnectDelay: 30000,
  requestTimeout: 30000,
};