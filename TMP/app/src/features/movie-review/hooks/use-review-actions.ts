import {
  CURRENT_USER_QUERY_KEY,
  REVIEW_QUERY_KEY,
} from "@/constants/query-keys";
import { SafeCurrentUser } from "@/types";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ReviewFormValues } from "../schemas";
import { SafeReview } from "../types";
import { useFetchReviews } from "./use-fetch-reviews";

export const useReviewActions = (movieId: number) => {
  "use memo";

  const queryClient = useQueryClient();
  const { data: reviews, isPending } = useFetchReviews();

  const add = useMutation({
    mutationFn: async (values: ReviewFormValues) => {
      const response = await fetch(`/api/movies/${movieId}/reviews`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rating: values.rating,
          watchDate: values.watchDate,
          note: values.note,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to save review");
      }

      return response.json();
    },
    onMutate: async (newReview) => {
      await Promise.all([
        queryClient.cancelQueries({ queryKey: REVIEW_QUERY_KEY }),
        queryClient.cancelQueries({ queryKey: CURRENT_USER_QUERY_KEY }),
      ]);

      const previousReviews =
        queryClient.getQueryData<SafeReview[]>(REVIEW_QUERY_KEY);
      const previousUser = queryClient.getQueryData<SafeCurrentUser>(
        CURRENT_USER_QUERY_KEY
      );

      const optimisticReview: SafeReview = {
        id: `temp-${Date.now()}`,
        movieId,
        rating: newReview.rating,
        note: newReview.note ?? null,
        watchedAt: new Date(newReview.watchDate).toISOString(),
        deletedAt: null,
        movie: {
          movieId,
          title: "",
          releaseDate: null,
        },
      };

      queryClient.setQueryData<SafeReview[]>(REVIEW_QUERY_KEY, (old = []) => {
        const filtered = old.filter((review) => review.movieId !== movieId);
        return [...filtered, optimisticReview];
      });

      if (previousUser) {
        const existingReview = previousReviews?.find(
          (review) => review.movieId === movieId
        );
        if (!existingReview || existingReview.deletedAt) {
          queryClient.setQueryData<SafeCurrentUser>(CURRENT_USER_QUERY_KEY, {
            ...previousUser,
            reviewCount: (previousUser.reviewCount || 0) + 1,
          });
        }
      }

      return { previousReviews, previousUser };
    },
    onError: (_err, _variables, context) => {
      if (context?.previousReviews) {
        queryClient.setQueryData(REVIEW_QUERY_KEY, context.previousReviews);
      }
      if (context?.previousUser) {
        queryClient.setQueryData(CURRENT_USER_QUERY_KEY, context.previousUser);
      }
      toast.error("Failed to review movie. Please try again.");
    },
    onSuccess: () => {
      toast.success("Movie reviewed successfully!");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: REVIEW_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
    },
  });

  const remove = useMutation({
    mutationFn: async () => {
      const response = await fetch(`/api/movies/${movieId}/reviews`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to remove review");
      }

      return response.json();
    },
    onMutate: async () => {
      await Promise.all([
        queryClient.cancelQueries({ queryKey: REVIEW_QUERY_KEY }),
        queryClient.cancelQueries({ queryKey: CURRENT_USER_QUERY_KEY }),
      ]);

      const previousReviews =
        queryClient.getQueryData<SafeReview[]>(REVIEW_QUERY_KEY);
      const previousUser = queryClient.getQueryData<SafeCurrentUser>(
        CURRENT_USER_QUERY_KEY
      );

      queryClient.setQueryData<SafeReview[]>(
        REVIEW_QUERY_KEY,
        (old = []) =>
          old?.map((review) => {
            if (review.movieId === movieId) {
              return {
                ...review,
                deletedAt: new Date().toISOString(),
              };
            }
            return review;
          }) ?? []
      );

      if (previousUser) {
        const existingReview = previousReviews?.find(
          (review) => review.movieId === movieId && !review.deletedAt
        );
        if (existingReview) {
          queryClient.setQueryData<SafeCurrentUser>(CURRENT_USER_QUERY_KEY, {
            ...previousUser,
            reviewCount: (previousUser.reviewCount || 0) - 1,
          });
        }
      }

      return { previousReviews, previousUser };
    },
    onError: (_err, _variables, context) => {
      if (context?.previousReviews) {
        queryClient.setQueryData(REVIEW_QUERY_KEY, context.previousReviews);
      }
      if (context?.previousUser) {
        queryClient.setQueryData(CURRENT_USER_QUERY_KEY, context.previousUser);
      }
      toast.error("Failed to remove review. Please try again.");
    },
    onSuccess: () => {
      toast.success("Review removed successfully");
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: REVIEW_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
    },
  });

  const review = reviews?.find(
    (review: SafeReview) => review.movieId === movieId
  );

  const isReviewed = !!review && !review.deletedAt;

  return {
    review,
    isReviewed,
    isPending: isPending || add.isPending || remove.isPending,
    addReview: (values: ReviewFormValues) => add.mutate(values),
    removeReview: () => remove.mutate(),
  };
};
