import { createSignal, createResource, createEffect } from 'solid-js';
import { apiClient } from '../../api/custom-client';

interface UseDocumentLoaderProps {
  documentStore: any;
}

interface DocumentLoader {
  loadDocument: (documentId: string) => Promise<void>;
  loadTextSpans: (documentId: string) => Promise<void>;
  loadEntities: (documentId: string) => Promise<void>;
  loading: () => boolean;
}

export function useDocumentLoader({ documentStore }: UseDocumentLoaderProps): DocumentLoader {
  const [loading, setLoading] = createSignal(false);

  const loadDocument = async (documentId: string) => {
    setLoading(true);
    documentStore.setLoading(true);
    documentStore.clearError();

    try {
      const contentResponse = await apiClient.GET('/api/v1/documents/{document_id}/content', {
        params: {
          path: { document_id: documentId },
          query: { format: 'json' }
        }
      });

      if (contentResponse) {
        documentStore.setDocument(documentId, contentResponse);
      } else {
        throw new Error('Failed to load document');
      }

      setLoading(false);
      documentStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load document:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      documentStore.setError(`Failed to load document: ${errorMessage}`);
      setLoading(false);
    }
  };

  const loadTextSpans = async (documentId: string) => {
    setLoading(true);
    documentStore.setLoading(true);

    try {
      const response = await apiClient.GET('/api/v1/documents/{document_id}/spans', {
        params: {
          path: { document_id: documentId },
          query: {
            confidence_threshold: 0.0,
            verified_only: false,
          }
        }
      });

      if (response && 'spans' in response) {
        documentStore.setTextSpans(response.spans || []);
      }

      setLoading(false);
      documentStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load text spans:', error);
      setLoading(false);
    }
  };

  const loadEntities = async (documentId: string) => {
    setLoading(true);
    documentStore.setLoading(true);

    try {
      const response = await apiClient.GET('/api/v1/documents/{document_id}/entities', {
        params: {
          path: { document_id: documentId },
          query: {
            min_confidence: 0.5,
            include_spans: true,
            limit: 100,
            offset: 0,
          }
        }
      });

      if (response && 'entities' in response) {
        documentStore.setEntities(response.entities || []);
      }

      setLoading(false);
      documentStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load entities:', error);
      setLoading(false);
    }
  };

  return {
    loadDocument,
    loadTextSpans,
    loadEntities,
    loading,
  };
}
