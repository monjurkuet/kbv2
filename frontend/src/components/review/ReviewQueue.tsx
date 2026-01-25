import { createMemo, For, Show, createEffect } from 'solid-js';
import { useReviewQueue } from '../../hooks/review/useReviewQueue';
import type { ReviewStoreType } from '../../stores/reviewStore';
import type { ReviewItem } from '../../stores/reviewStore';

interface ReviewQueueProps {
  reviewStore: ReviewStoreType;
}

const ReviewQueue = (props: ReviewQueueProps) => {
  const { loadReviewQueue, loading } = useReviewQueue({ reviewStore: props.reviewStore });

  createEffect(() => {
    loadReviewQueue();
  });

  const filteredReviews = createMemo(() => {
    const reviews = props.reviewStore.reviewStore.reviewQueue;
    const filters = props.reviewStore.reviewStore.filters;

    return reviews.filter((review) => {
      if (filters.type !== 'all' && review.item_type !== filters.type) return false;
      if (review.priority < filters.priority) return false;
      if (review.confidence_score !== undefined && review.confidence_score < filters.confidence) return false;
      if (filters.status !== 'all' && review.status !== filters.status) return false;
      return true;
    });
  });

  const itemTypeName = (review: ReviewItem): string => {
    const typeMap: Record<string, string> = {
      entity_resolution: 'Entity Resolution',
      edge_validation: 'Edge Validation',
      merged_entity: 'Merged Entity',
    };
    return typeMap[review.item_type] || review.item_type;
  };

  const priorityColor = (priority: number): string => {
    if (priority >= 8) return 'bg-red-100 text-red-800 border-red-200';
    if (priority >= 5) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-yellow-100 text-yellow-800 border-yellow-200';
  };

  const statusColor = (status: string): string => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-800 border-green-200';
      case 'rejected': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-blue-100 text-blue-800 border-blue-200';
    }
  };

  return (
    <div class="ReviewQueue flex flex-col h-full">
      <div class="bg-white border-b border-gray-200 p-4">
        <h2 class="text-2xl font-bold text-gray-900">Review Queue</h2>
        <p class="text-sm text-gray-600">Human-in-the-loop entity resolution</p>
      </div>

      <div class="flex-1 overflow-auto">
        <Show
          when={filteredReviews().length > 0}
          fallback={
            <div class="p-8 text-center text-gray-500">
              <div class="text-5xl mb-4">📋</div>
              <div class="text-lg font-medium">No pending reviews</div>
              <p class="text-sm mt-1">All reviews have been processed or the queue is empty</p>
            </div>
          }
        >
          <div class="divide-y divide-gray-200">
            <For each={filteredReviews()}>
              {(review) => (
                <div
                  class="p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                  onClick={() => props.reviewStore.setActiveReview(review)}
                >
                  <div class="flex justify-between items-start">
                    <div class="flex-1">
                      <div class="flex items-center space-x-3 mb-2">
                        <h3 class="font-semibold text-gray-900">{review.entity_id || review.edge_id || review.merged_entity_ids?.join(', ')}</h3>
                        <span class={`text-xs px-2 py-1 rounded-full border ${itemTypeName(review) === 'Entity Resolution' ? 'bg-blue-100 text-blue-800 border-blue-200' : 'bg-purple-100 text-purple-800 border-purple-200'}`}>
                          {itemTypeName(review)}
                        </span>
                      </div>

                      <div class="space-y-1 text-sm text-gray-600">
                        <div class="flex items-center space-x-4">
                          <span class={`text-xs px-2 py-1 rounded-full border ${priorityColor(review.priority)}`}>
                            Priority: {review.priority}
                          </span>
                          <span class={`text-xs px-2 py-1 rounded-full border ${statusColor(review.status)}`}>
                            {review.status.charAt(0).toUpperCase() + review.status.slice(1)}
                          </span>
                        </div>

                        <Show when={review.confidence_score !== undefined}>
                          <div class="flex items-center space-x-2">
                            <span class="text-xs font-medium">Confidence:</span>
                            <div class="flex-1 bg-gray-200 rounded-full h-2">
                              <div
                                class="bg-blue-500 h-2 rounded-full"
                                style={{ width: `${(review.confidence_score || 0) * 100}%` }}
                              />
                            </div>
                            <span class="text-xs font-mono">{Math.round((review.confidence_score || 0) * 100)}%</span>
                          </div>
                        </Show>

                        <Show when={review.grounding_quote}>
                          <div class="mt-1 p-2 bg-gray-100 rounded text-xs text-gray-700 line-clamp-2">
                            "{review.grounding_quote}"
                          </div>
                        </Show>

                        <Show when={review.source_text}>
                          <div class="mt-1 p-2 bg-gray-100 rounded text-xs text-gray-700 line-clamp-2">
                            Source: {review.source_text}
                          </div>
                        </Show>

                        <div class="mt-1 text-xs text-gray-500">
                          Created {new Date(review.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </For>
          </div>
        </Show>
      </div>
    </div>
  );
};

export default ReviewQueue;