export interface GraphNode {
  key: string;
  label: string;
  attributes: {
    x?: number;
    y?: number;
    size?: number;
    community_id?: string;
    entity_count?: number;
    confidence?: number;
    type?: string;
    [key: string]: any;
  };
}

export interface GraphEdge {
  source: string;
  target: string;
  attributes: {
    weight?: number;
    confidence?: number;
    [key: string]: any;
  };
}

export interface GraphSummary {
  nodes: GraphNode[];
  edges: GraphEdge[];
  [key: string]: any;
}