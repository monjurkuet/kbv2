import { createStore } from 'solid-js/store';
import type { DocumentContentResponse, DocumentEntityResponse } from '../api';

interface TextSpan {
  start_offset: number;
  end_offset: number;
  entity_type: string;
  confidence: number;
  entity_name: string;
  entity_id: string;
}

export interface DocumentStore {
  currentDocumentId: string | null;
  documentContent: DocumentContentResponse | null;
  textSpans: TextSpan[];
  entities: DocumentEntityResponse[];
  activeEntity: string | null;
  loading: boolean;
  error: string | null;
  scrollToOffset: number | null;
}

const [documentStore, setDocumentStore] = createStore<DocumentStore>({
  currentDocumentId: null,
  documentContent: null,
  textSpans: [],
  entities: [],
  activeEntity: null,
  loading: false,
  error: null,
  scrollToOffset: null,
});

export const createDocumentStore = () => {
  const setDocument = (documentId: string, content: DocumentContentResponse) => {
    setDocumentStore({
      currentDocumentId: documentId,
      documentContent: content,
      loading: false,
      error: null,
    });
  };

  const setActiveEntity = (entityId: string | null) => {
    setDocumentStore('activeEntity', entityId);
  };

  const setTextSpans = (spans: TextSpan[]) => {
    setDocumentStore('textSpans', spans);
  };

  const setEntities = (entities: DocumentEntityResponse[]) => {
    setDocumentStore('entities', entities);
  };

  const setScrollToOffset = (offset: number | null) => {
    setDocumentStore('scrollToOffset', offset);
  };

  const clearError = () => {
    setDocumentStore('error', null);
  };

  const setLoading = (loading: boolean) => {
    setDocumentStore('loading', loading);
  };

  const setError = (error: string) => {
    setDocumentStore({
      error,
      loading: false,
    });
  };

  return {
    documentStore,
    setDocument,
    setActiveEntity,
    setTextSpans,
    setEntities,
    setScrollToOffset,
    clearError,
    setLoading,
    setError,
  };
};

export type DocumentStoreType = ReturnType<typeof createDocumentStore>;
