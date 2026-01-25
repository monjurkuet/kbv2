export const ENTITY_COLORS = {
  community: '#4CAF50',
  center: '#2196F3',
  neighbor: '#757575',
  edge: 'rgba(0, 0, 0, 0.3)',
} as const;

export const EDGE_COLORS = {
  default: 'rgba(0, 0, 0, 0.3)',
  lowConfidence: 'rgba(0, 0, 0, 0.1)',
  highConfidence: 'rgba(0, 0, 0, 0.5)',
} as const;

export const ENTITY_TYPES: Record<string, string> = {
  person: '#3b82f6',         // Blue
  organization: '#10b981',   // Green
  location: '#f59e0b',       // Yellow
  event: '#ef4444',          // Red
  concept: '#8b5cf6',        // Purple
  artifact: '#6b7280',       // Gray
};

export const NODE_SIZES = {
  min: 4,
  max: 20,
  community: 24,
} as const;

export interface SigmaSettings {
  renderer: {
    backgroundColor: string;
    labelColor: string;
    defaultNodeColor: string;
    defaultEdgeColor: string;
  };
  forceAtlas: {
    gravity: number;
    scalingRatio: number;
    slowDown: number;
    strongGravityMode: boolean;
  };
}

export const SIGMA_SETTINGS: SigmaSettings = {
  renderer: {
    backgroundColor: '#ffffff',
    labelColor: '#374151',
    defaultNodeColor: '#4CAF50',
    defaultEdgeColor: 'rgba(0, 0, 0, 0.3)',
  },
  forceAtlas: {
    gravity: 0.1,
    scalingRatio: 10,
    slowDown: 1,
    strongGravityMode: false,
  },
};
