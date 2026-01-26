import { createSignal, createResource, createEffect } from 'solid-js';
import { getPendingReviewsApiV1ReviewPendingGet, getReviewApiV1ReviewReviewIdGet, approveReviewApiV1ReviewReviewIdApprovePost, rejectReviewApiV1ReviewReviewIdRejectPost } from '../../api/sdk.gen';
import type { ReviewStoreType } from '../../stores/reviewStore';

interface UseReviewQueueProps {
  reviewStore: ReviewStoreType;
}

interface ReviewQueueLoader {
  loadReviewQueue: (limit?: number, offset?: number) => Promise<void>;
  loadReview: (reviewId: string) => Promise<void>;
  approveReview: (reviewId: string, notes?: string) => Promise<boolean>;
  rejectReview: (reviewId: string, corrections?: object, notes?: string) => Promise<boolean>;
  editReview: (reviewId: string, updates: object) => Promise<boolean>;
  loading: () => boolean;
}

export function useReviewQueue({ reviewStore }: UseReviewQueueProps): ReviewQueueLoader {
  const [loading, setLoading] = createSignal(false);

  const loadReviewQueue = async (limit = 50, offset = 0) => {
    setLoading(true);
    reviewStore.setLoading(true);
    reviewStore.clearError();

    try {
      const response = await getPendingReviewsApiV1ReviewPendingGet({
        params: {
          query: {
            limit,
            offset,
          }
        }
      });

      if (response.data && Array.isArray(response.data)) {
        reviewStore.setReviewQueue(response.data, offset > 0 ? offset : 0);
      }

      setLoading(false);
      reviewStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load review queue:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      reviewStore.setError(`Failed to load review queue: ${errorMessage}`);
      setLoading(false);
    }
  };

  const loadReview = async (reviewId: string) => {
    setLoading(true);
    reviewStore.setLoading(true);

    try {
      const response = await getReviewApiV1ReviewReviewIdGet({
        params: {
          path: { review_id: reviewId }
        }
      });

      if (response.data) {
        reviewStore.setActiveReview(response.data);
      }

      setLoading(false);
      reviewStore.setLoading(false);
    } catch (error) {
      console.error('Failed to load review:', error);
      setLoading(false);
    }
  };

  const approveReview = async (reviewId: string, notes?: string) => {
    setLoading(true);

    try {
      const body = notes ? { reviewer_notes: notes } : {};
      const response = await approveReviewApiV1ReviewReviewIdApprovePost({
        params: {
          path: { review_id: reviewId }
        },
        body,
      });

      if (response.data) {
        reviewStore.updateReview(reviewId, { 
          status: 'approved', 
          reviewed_at: new Date().toISOString(),
          reviewer_notes: notes 
        });
        return true;
      }

      setLoading(false);
      return false;
    } catch (error) {
      console.error('Failed to approve review:', error);
      setLoading(false);
      return false;
    }
  };

  const rejectReview = async (reviewId: string, corrections?: object, notes?: string) => {
    setLoading(true);

    try {
      const body = corrections ? { corrections, reviewer_notes: notes } : { reviewer_notes: notes };
      const response = await rejectReviewApiV1ReviewReviewIdRejectPost({
        params: {
          path: { review_id: reviewId }
        },
        body,
      });

      if (response.data) {
        reviewStore.updateReview(reviewId, { 
          status: 'rejected', 
          reviewed_at: new Date().toISOString(),
          reviewer_notes: notes 
        });
        return true;
      }

      setLoading(false);
      return false;
    } catch (error) {
      console.error('Failed to reject review:', error);
      setLoading(false);
      return false;
    }
  };

  const editReview = async (reviewId: string, updates: object) => {
    console.warn('Edit review not implemented - no API endpoint available');
    return false;
  };

  return {
    loadReviewQueue,
    loadReview,
    approveReview,
    rejectReview,
    editReview,
    loading,
  };
}