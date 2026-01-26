import { createSignal, createResource, createEffect } from 'solid-js';
import { getDocumentContentApiV1DocumentsDocumentIdContentGet, getDocumentSpansApiV1DocumentsDocumentIdSpansGet, getDocumentEntitiesApiV1DocumentsDocumentIdEntitiesGet } from '../../api/sdk.gen';

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
      const contentResponse = await getDocumentContentApiV1DocumentsDocumentIdContentGet({
        params: {
          path: { document_id: documentId },
          query: { format: 'json' }
        }
      });

      if (contentResponse.data) {
        documentStore.setDocument(documentId, contentResponse.data);
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
      const response = await getDocumentSpansApiV1DocumentsDocumentIdSpansGet({
        params: {
          path: { document_id: documentId },
          query: {
            confidence_threshold: 0.0,
            verified_only: false,
          }
        }
      });

      if (response.data && 'spans' in response.data) {
        documentStore.setTextSpans(response.data.spans || []);
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
      const response = await getDocumentEntitiesApiV1DocumentsDocumentIdEntitiesGet({
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

      if (response.data && 'entities' in response.data) {
        documentStore.setEntities(response.data.entities || []);
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
