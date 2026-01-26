import { createEffect, createMemo, createSignal, on, Show } from 'solid-js';
import { useDocumentLoader } from '../../hooks/document/useDocumentLoader';
import type { DocumentStoreType } from '../../stores/documentStore';

interface TextSpan {
  start_offset: number;
  end_offset: number;
  entity_type: string;
  confidence: number;
  entity_name: string;
  entity_id: string;
}

interface DocumentViewerProps {
  documentStore: DocumentStoreType;
  documentId: string;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function applyHighlights(text: string, spans: TextSpan[]): string {
  if (!spans || spans.length === 0) {
    return escapeHtml(text);
  }

  const sortedSpans = [...spans].sort((a, b) => {
    if (a.start_offset !== b.start_offset) {
      return a.start_offset - b.start_offset;
    }
    return (b.end_offset - b.start_offset) - (a.end_offset - a.start_offset);
  });

  const result: string[] = [];
  let lastIndex = 0;

  for (const span of sortedSpans) {
    if (span.start_offset > text.length) break;

    if (span.start_offset > lastIndex) {
      result.push(escapeHtml(text.slice(lastIndex, span.start_offset)));
    }

    const highlightedText = escapeHtml(
      text.slice(span.start_offset, Math.min(span.end_offset, text.length))
    );
    
    const entityType = span.entity_type || 'default';
    const confidence = span.confidence || 0;
    const title = `${span.entity_name || 'Entity'} (${entityType}) - ${Math.round(confidence * 100)}% confidence`;
    
    result.push(
      `<mark class="entity-highlight entity-${entityType}" ` +
      `data-entity-id="${span.entity_id || ''}" ` +
      `data-entity-type="${entityType}" ` +
      `data-confidence="${confidence}" ` +
      `style="cursor: pointer; pointer-events: auto;" ` +
      `title="${title}">${highlightedText}</mark>`
    );

    lastIndex = Math.max(lastIndex, span.end_offset);
  }

  if (lastIndex < text.length) {
    result.push(escapeHtml(text.slice(lastIndex)));
  }

  return result.join('');
}

const DocumentViewer = (props: DocumentViewerProps) => {
  const { loadDocument, loadTextSpans, loadEntities } = useDocumentLoader({
    documentStore: props.documentStore
  });

  const [containerRef, setContainerRef] = createSignal<HTMLDivElement | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  createEffect(() => {
    setError(null);
    loadDocument(props.documentId).catch(err => {
      setError(`Failed to load document: ${err instanceof Error ? err.message : 'Unknown error'}`);
    });
    loadTextSpans(props.documentId).catch(err => {
      console.error('Failed to load text spans:', err);
    });
    loadEntities(props.documentId).catch(err => {
      console.error('Failed to load entities:', err);
    });
  });

  createEffect(() => {
    const offset = props.documentStore.documentStore.scrollToOffset;
    const container = containerRef();
    
    if (offset !== null && container) {
      const textContent = container.textContent || '';
      const approximateLine = Math.floor(offset / 80);
      const lineHeight = 20;
      const scrollPosition = approximateLine * lineHeight;
      
      container.scrollTo({
        top: scrollPosition,
        behavior: 'smooth'
      });
      
      props.documentStore.setScrollToOffset(null);
    }
  });

  const highlightedContent = createMemo(() => {
    const content = props.documentStore.documentStore.documentContent?.content || '';
    const spans = props.documentStore.documentStore.textSpans || [];
    return applyHighlights(content, spans);
  });

  const handleEntityClick = (entityId: string) => {
    props.documentStore.setActiveEntity(entityId);
    
    const spans = props.documentStore.documentStore.textSpans;
    const entitySpan = spans.find(span => span.entity_id === entityId);
    
    if (entitySpan) {
      props.documentStore.setScrollToOffset(entitySpan.start_offset);
    }
  };

  return (
    <div class="flex h-full bg-white">
      <div class="flex-1 p-6 overflow-auto" ref={setContainerRef}>
        <Show when={error()}>
          <div class="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md mb-4">
            {error()}
          </div>
        </Show>
        <Show
          when={!props.documentStore.documentStore.loading && props.documentStore.documentStore.documentContent}
          fallback={<div class="text-center text-gray-500 py-8" data-testid="loading-indicator">Loading document...</div>}
        >
          <pre 
            class="whitespace-pre-wrap font-mono text-sm leading-relaxed text-gray-800"
            innerHTML={highlightedContent()}
            onClick={(e) => {
              const target = e.target as HTMLElement;
              if (target.tagName === 'MARK' && target.dataset.entityId) {
                handleEntityClick(target.dataset.entityId);
              }
            }}
          />
        </Show>
      </div>

      <DocumentSidebar 
        documentStore={props.documentStore}
        onEntityClick={handleEntityClick}
      />
    </div>
  );
};

interface DocumentSidebarProps {
  documentStore: DocumentStoreType;
  onEntityClick: (entityId: string) => void;
}

const DocumentSidebar = (props: DocumentSidebarProps) => {
  const entities = createMemo(() => props.documentStore.documentStore.entities || []);

  const groupedEntities = createMemo(() => {
    const groups: Record<string, any[]> = {};
    entities().forEach(entity => {
      const type = entity.entity_type || 'unknown';
      if (!groups[type]) groups[type] = [];
      groups[type].push(entity);
    });
    return groups;
  });

  return (
    <div class="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
      <h3 class="text-lg font-semibold mb-4 text-gray-900">Extracted Entities</h3>
      
      <Show
        when={entities().length > 0}
        fallback={<div class="text-sm text-gray-500">No entities found</div>}
      >
        <div class="space-y-4">
          {Object.entries(groupedEntities()).map(([type, entitiesList]) => (
            <div key={type}>
              <h4 class="text-xs font-semibold text-gray-600 uppercase tracking-wide mb-2">
                {type} ({entitiesList.length})
              </h4>
              <div class="space-y-1">
                {entitiesList.map((entity) => (
                  <div
                    key={entity.entity_id}
                    class={`p-2 rounded cursor-pointer transition-colors ${
                      props.documentStore.documentStore.activeEntity === entity.entity_id
                        ? 'bg-blue-100 border border-blue-300'
                        : 'bg-white hover:bg-gray-100 border border-gray-200'
                    }`}
                    onClick={() => props.onEntityClick(entity.entity_id)}
                  >
                    <div class="font-medium text-sm text-gray-900">
                      {entity.name}
                    </div>
                    <div class="flex items-center justify-between mt-1">
                      <span class={`text-xs px-2 py-0.5 rounded-full ${
                        entity.confidence > 0.8
                          ? 'bg-green-100 text-green-800'
                          : entity.confidence > 0.6
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {Math.round(entity.confidence * 100)}%
                      </span>
                      <span class="text-xs text-gray-500 font-mono">
                        {entity.entity_id?.slice(0, 8)}...
                      </span>
                    </div>
                    {entity.description && (
                      <div class="text-xs text-gray-600 mt-1 line-clamp-2">
                        {entity.description}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Show>
    </div>
  );
};

export default DocumentViewer;