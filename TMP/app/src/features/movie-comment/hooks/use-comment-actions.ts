import { COMMENTS_QUERY_KEY } from "@/constants/query-keys";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { CommentEditApiValues } from "../schemas";
import { CommentPage } from "../types";

interface CommentInput {
  content: string;
  depth: number;
  isSpoiler: boolean;
  parentId: string | null;
}

export const useCommentActions = (movieId: number) => {
  const queryClient = useQueryClient();

  const create = useMutation({
    mutationFn: async (data: CommentInput) => {
      const response = await fetch(`/api/movies/${movieId}/comments`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error("Failed to post comment");
      }
      return response.json();
    },
    onError: () => {
      toast.error("Failed to post comment. Please try again.");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });
      toast.success("Comment posted successfully");
    },
  });

  const remove = useMutation({
    mutationFn: async (commentId: string) => {
      const response = await fetch(
        `/api/movies/${movieId}/comments/${commentId}`,
        {
          method: "DELETE",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to delete comment");
      }

      return response.json();
    },
    onError: () => {
      toast.error("Failed to remove comment. Please try again.");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });
      toast.success("Comment deleted successfully");
    },
  });

  const edit = useMutation({
    mutationFn: async (values: CommentEditApiValues) => {
      const response = await fetch(
        `/api/movies/${movieId}/comments/${values.commentId}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            content: values.content,
            isSpoiler: values.isSpoiler,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to edit comment");
      }

      return response.json();
    },
    onMutate: async (data) => {
      const queryKeys = queryClient.getQueriesData<{
        pages: CommentPage[];
        pageParams: (string | undefined)[];
      }>({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });

      queryKeys.forEach(([queryKey, oldData]) => {
        if (!oldData) return;

        queryClient.setQueryData(queryKey, {
          ...oldData,
          pages: oldData.pages.map((page) => ({
            ...page,
            comments: page.comments.map((comment) => {
              if (comment.id === data.commentId) {
                return {
                  ...comment,
                  content: data.content,
                  isSpoiler: data.isSpoiler,
                  isEdited: true,
                };
              }
              return comment;
            }),
          })),
        });
      });
    },
    onError: () => {
      queryClient.invalidateQueries({
        queryKey: COMMENTS_QUERY_KEY(movieId),
      });
      toast.error("Failed to edit comment. Please try again.");
    },
    onSuccess: () => {
      toast.success("Comment updated successfully");
    },
  });

  return {
    isPending: create.isPending || remove.isPending || edit.isPending,
    createComment: (data: CommentInput) => create.mutate(data),
    removeComment: (commentId: string, options: { onSuccess: () => void }) => {
      remove.mutate(commentId, {
        onSuccess: () => {
          options.onSuccess();
        },
      });
    },
    editComment: (
      values: CommentEditApiValues,
      options: { onSuccess: () => void }
    ) => {
      edit.mutate(values, {
        onSuccess: () => {
          options.onSuccess();
        },
      });
    },
  };
};
