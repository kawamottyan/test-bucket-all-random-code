import { COMMENTS_QUERY_KEY } from "@/constants/query-keys";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { CommentPage } from "../types";

export const useCommentFavoriteActions = (movieId: number) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({
      commentId,
    }: {
      commentId: string;
      isFavorite: boolean;
    }) => {
      const response = await fetch(
        `/api/movies/${movieId}/comments/${commentId}/favorite`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to favorite comment");
      }

      return response.json();
    },
    onMutate: async ({ commentId, isFavorite }) => {
      await queryClient.cancelQueries({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });

      const previousComments = queryClient.getQueriesData<{
        pages: CommentPage[];
        pageParams: (string | undefined)[];
      }>({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });

      previousComments.forEach(([queryKey, data]) => {
        if (!data) return;

        queryClient.setQueryData(queryKey, {
          pages: data.pages.map((page) => ({
            ...page,
            comments: page.comments.map((comment) => {
              if (comment.id === commentId) {
                return {
                  ...comment,
                  isFavorite: !isFavorite,
                  favoriteCount: isFavorite
                    ? comment.favoriteCount - 1
                    : comment.favoriteCount + 1,
                };
              }
              return comment;
            }),
          })),
          pageParams: data.pageParams,
        });
      });

      return { previousComments };
    },
    onError: (_, _variables, context) => {
      if (context?.previousComments) {
        context.previousComments.forEach(([queryKey, data]) => {
          queryClient.setQueryData(queryKey, data);
        });
      }
      toast.error("Failed to favorite comment. Please try again.");
    },
    onSuccess: (response) => {
      const message = response.data.favorited
        ? "Added to favorites"
        : "Removed from favorites";
      toast.success(message);
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });
    },
  });
};
