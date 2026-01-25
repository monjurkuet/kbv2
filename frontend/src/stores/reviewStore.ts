import { createStore } from 'solid-js/store';

export interface ReviewItem {
  id: string;
  item_type: 'entity_resolution' | 'edge_validation' | 'merged_entity';
  entity_id?: string;
  edge_id?: string;
  document_id?: string;
  merged_entity_ids?: string[];
  confidence_score?: number;
  grounding_quote?: string;
  source_text?: string;
  status: 'pending' | 'approved' | 'rejected';
  priority: number;
  created_at: string;
  reviewed_at?: string;
  reviewer_notes?: string;
}

export interface ReviewQueueState {
  reviewQueue: ReviewItem[];
  activeReview: ReviewItem | null;
  hasMore: boolean;
  page: number;
  total: number;
  loading: boolean;
  error: string | null;
  filters: {
    type: 'all' | 'entity_resolution' | 'edge_validation' | 'merged_entity';
    priority: number;
    confidence: number;
    status: 'all' | 'pending' | 'approved' | 'rejected';
  };
}

const [reviewStore, setReviewStore] = createStore<ReviewQueueState>({
  reviewQueue: [],
  activeReview: null,
  hasMore: true,
  page: 0,
  total: 0,
  loading: false,
  error: null,
  filters: {
    type: 'all',
    priority: 0,
    confidence: 0,
    status: 'all',
  },
});

export const createReviewStore = () => {
  const setReviewQueue = (reviews: ReviewItem[], page: number = 0) => {
    setReviewStore({
      reviewQueue: page === 0 ? reviews : [...reviewStore.reviewQueue, ...reviews],
      hasMore: reviews.length > 0,
      page,
    });
  };

  const setActiveReview = (review: ReviewItem | null) => {
    setReviewStore('activeReview', review);
  };

  const addReview = (review: ReviewItem) => {
    setReviewStore('reviewQueue', (prev) => [review, ...prev]);
  };

  const updateReview = (id: string, updates: Partial<ReviewItem>) => {
    setReviewStore('reviewQueue', (prev) => 
      prev.map(r => r.id === id ? { ...r, ...updates } : r)
    );
  };

  const removeReview = (id: string) => {
    setReviewStore('reviewQueue', (prev) => prev.filter(r => r.id !== id));
  };

  const clearError = () => {
    setReviewStore('error', null);
  };

  const setLoading = (loading: boolean) => {
    setReviewStore('loading', loading);
  };

  const setError = (error: string) => {
    setReviewStore({
      error,
      loading: false,
    });
  };

  const setFilters = (filters: Partial<ReviewQueueState['filters']>) => {
    setReviewStore('filters', (prev) => ({ ...prev, ...filters }));
  };

  return {
    reviewStore,
    setReviewQueue,
    setActiveReview,
    addReview,
    updateReview,
    removeReview,
    clearError,
    setLoading,
    setError,
    setFilters,
  };
};

export type ReviewStoreType = ReturnType<typeof createReviewStore>;