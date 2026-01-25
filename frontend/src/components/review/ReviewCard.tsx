import { createSignal, Show, Switch, Match } from 'solid-js';
import { useReviewQueue } from '../../hooks/review/useReviewQueue';
import type { ReviewStoreType } from '../../stores/reviewStore';

interface ReviewCardProps {
  reviewStore: ReviewStoreType;
}

const ReviewCard = (props: ReviewCardProps) => {
  const { approveReview, rejectReview, editReview, loading } = useReviewQueue({ reviewStore: props.reviewStore });
  const [showEdit, setShowEdit] = createSignal(false);
  const [editData, setEditData] = createSignal({});
  const [notes, setNotes] = createSignal('');

  const activeReview = () => props.reviewStore.activeReview;

  const handleApprove = async () => {
    if (activeReview()) {
      await approveReview(activeReview()!.id, notes());
      setShowEdit(false);
      setNotes('');
    }
  };

  const handleReject = async () => {
    if (activeReview()) {
      await rejectReview(activeReview()!.id, activeReview()!.item_type === 'entity_resolution' ? { name: 'Test' } : undefined, notes());
      setShowEdit(false);
      setNotes('');
    }
  };

  const handleEdit = async () => {
    if (activeReview()) {
      await editReview(activeReview()!.id, editData());
      setShowEdit(false);
    }
  };

  const ReviewTypeDetail = () => {
    if (!activeReview()) return null;

    const review = activeReview()!;
    const typeName = review.item_type === 'entity_resolution' ? 'Entity Resolution' :
                   review.item_type === 'edge_validation' ? 'Edge Validation' :
                   'Merged Entity';

    return (
      <div class="space-y-6">
        <div class="bg-gray-50 rounded-lg p-4">
          <h3 class="text-lg font-semibold text-gray-900 mb-3">{typeName}</h3>
          
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Entity ID</label>
              <input
                type="text"
                value={review.entity_id || ''}
                readonly
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-gray-100"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <input
                type="text"
                value={review.status}
                readonly
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-gray-100"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Priority</label>
              <input
                type="text"
                value={review.priority}
                readonly
                class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md bg-gray-100"
              />
            </div>
            <div>
              <label class="block text-sm font-medium text-gray-700 mb-1">Confidence</label>
              <div class="mt-1">
                <div class="flex items-center space-x-2">
                  <div class="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      class="bg-blue-500 h-2 rounded-full"
                      style={{ width: `${(review.confidence_score || 0) * 100}%` }}
                    />
                  </div>
                  <span class="text-xs font-mono">{Math.round((review.confidence_score || 0) * 100)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <Show when={review.grounding_quote}>
          <div class="bg-green-50 border border-green-100 rounded-lg p-4">
            <h4 class="text-sm font-semibold text-green-800 mb-2">Grounding Quote</h4>
            <div class="text-sm text-green-900 p-2 bg-white rounded border border-green-200">
              "{review.grounding_quote}"
            </div>
          </div>
        </Show>

        <Show when={review.source_text}>
          <div class="bg-blue-50 border border-blue-100 rounded-lg p-4">
            <h4 class="text-sm font-semibold text-blue-800 mb-2">Source Text</h4>
            <div class="text-sm text-blue-900 p-2 bg-white rounded border border-blue-200">
              <pre class="whitespace-pre-wrap text-xs">{review.source_text}</pre>
            </div>
          </div>
        </Show>

        <Show when={review.created_at}>
          <div class="flex justify-end text-xs text-gray-500">
            Created: {new Date(review.created_at).toLocaleString()}
          </div>
        </Show>
      </div>
    );
  };

  const ReviewActions = () => {
    if (!activeReview()) return null;

    return (
      <div class="flex flex-col space-y-4">
        <div class="bg-gray-50 rounded-lg p-4">
          <label class="block text-sm font-medium text-gray-700 mb-2">
            Reviewer Notes (Optional)
          </label>
          <textarea
            value={notes()}
            onInput={(e) => setNotes(e.currentTarget.value)}
            placeholder="Add any notes about your decision..."
            rows={3}
            class="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div class="flex space-x-3">
          <button
            onClick={handleApprove}
            disabled={loading()}
            class="flex-1 px-4 py-3 text-sm font-semibold text-white bg-green-600 hover:bg-green-700 disabled:bg-green-300 rounded-lg transition-colors"
          >
            {loading() ? 'Processing...' : 'Approve Merge'}
          </button>

          <button
            onClick={handleReject}
            disabled={loading()}
            class="flex-1 px-4 py-3 text-sm font-semibold text-white bg-red-600 hover:bg-red-700 disabled:bg-red-300 rounded-lg transition-colors"
          >
            {loading() ? 'Processing...' : 'Reject'}
          </button>

          <button
            onClick={() => setShowEdit(!showEdit())}
            disabled={loading()}
            class="px-4 py-3 text-sm font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 rounded-lg transition-colors"
          >
            {showEdit() ? 'Cancel' : 'Edit Metadata'}
          </button>
        </div>

        <Show when={showEdit()}>
          <div class="bg-yellow-50 border border-yellow-100 rounded-lg p-4">
            <h4 class="text-sm font-semibold text-yellow-800 mb-3">Edit Metadata</h4>
            
            <div class="space-y-3">
              <div>
                <label class="block text-xs font-medium text-yellow-700 mb-1">
                  Entity Name
                </label>
                <input
                  type="text"
                  value={editData().name || ''}
                  onInput={(e) => setEditData({ ...editData(), name: e.currentTarget.value })}
                  placeholder="New entity name"
                  class="w-full px-3 py-2 text-sm border border-yellow-200 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-yellow-500"
                />
              </div>

              <div>
                <label class="block text-xs font-medium text-yellow-700 mb-1">
                  Entity Type
                </label>
                <input
                  type="text"
                  value={editData().entity_type || ''}
                  onInput={(e) => setEditData({ ...editData(), entity_type: e.currentTarget.value })}
                  placeholder="New entity type"
                  class="w-full px-3 py-2 text-sm border border-yellow-200 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-yellow-500"
                />
              </div>

              <div>
                <label class="block text-xs font-medium text-yellow-700 mb-1">
                  Description
                </label>
                <textarea
                  value={editData().description || ''}
                  onInput={(e) => setEditData({ ...editData(), description: e.currentTarget.value })}
                  placeholder="New description..."
                  rows={2}
                  class="w-full px-3 py-2 text-sm border border-yellow-200 rounded-md bg-white focus:outline-none focus:ring-1 focus:ring-yellow-500"
                />
              </div>
            </div>

            <div class="mt-4 flex space-x-3">
              <button
                onClick={handleEdit}
                disabled={loading()}
                class="flex-1 px-4 py-2 text-sm font-semibold text-white bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-300 rounded-md"
              >
                Save Changes
              </button>
              <button
                onClick={() => {
                  setShowEdit(false);
                  setEditData({});
                }}
                class="flex-1 px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-md"
              >
                Cancel
              </button>
            </div>
          </div>
        </Show>
      </div>
    );
  };

  return (
    <div class="flex-1 flex flex-col min-h-0">
      <div class="bg-white border-b border-gray-200 p-4">
        <div class="flex items-center justify-between">
          <h2 class="text-2xl font-bold text-gray-900">Review Queue</h2>
          <button
            onClick={() => props.reviewStore.setActiveReview(null)}
            class="text-gray-400 hover:text-gray-600 font-semibold"
          >
            &larr; Back to Queue
          </button>
        </div>
        <p class="text-sm text-gray-600">Human-in-the-loop entity resolution</p>
      </div>

      <div class="flex-1 flex flex-col md:flex-row min-h-0">
        <div class="w-full md:w-1/2 p-6">
          <Show when={activeReview()} fallback={
            <div class="flex items-center justify-center h-full text-gray-400">
              <div class="text-center">
                <div class="text-5xl mb-4">📋</div>
                <div class="text-lg">No Review Selected</div>
                <p class="text-sm">Select a review from the queue to examine details</p>
              </div>
            </div>
          }>
            <ReviewTypeDetail />
          </Show>
        </div>

        <div class="w-full md:w-1/2 bg-gray-50 border-l border-gray-200 p-6">
          <Show when={activeReview()} fallback={
            <div class="flex items-center justify-center h-full text-gray-400">
              <div class="text-center">
                <div class="text-5xl mb-4">📝</div>
                <div class="text-lg">No Review Selected</div>
                <p class="text-sm">Select a review to take action</p>
              </div>
            </div>
          }>
            <ReviewActions />
          </Show>
        </div>
      </div>
    </div>
  );
};

export default ReviewCard;