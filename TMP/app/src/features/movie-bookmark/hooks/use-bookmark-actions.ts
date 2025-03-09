import {
  BOOKMARK_QUERY_KEY,
  CURRENT_USER_QUERY_KEY,
} from "@/constants/query-keys";
import { SafeCurrentUser } from "@/types";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { SafeBookmark } from "../types";
import { useFetchBookmarks } from "./use-fetch-bookmarks";

export const useBookmarkActions = (movieId: number) => {
  "use memo";
  const queryClient = useQueryClient();
  const { data: bookmarks, isPending } = useFetchBookmarks();

  const toggleBookmark = useMutation({
    mutationFn: async () => {
      const response = await fetch(`/api/movies/${movieId}/bookmarks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      if (!response.ok) {
        throw new Error("Failed to toggle bookmark");
      }
      return response.json();
    },
    onMutate: async () => {
      await Promise.all([
        queryClient.cancelQueries({ queryKey: BOOKMARK_QUERY_KEY }),
        queryClient.cancelQueries({ queryKey: CURRENT_USER_QUERY_KEY }),
      ]);

      const previousBookmarks =
        queryClient.getQueryData<SafeBookmark[]>(BOOKMARK_QUERY_KEY);
      const previousUser = queryClient.getQueryData<SafeCurrentUser>(
        CURRENT_USER_QUERY_KEY
      );

      const existingBookmark = previousBookmarks?.find(
        (bookmark) => bookmark.movieId === movieId
      );

      queryClient.setQueryData<SafeBookmark[]>(
        BOOKMARK_QUERY_KEY,
        (old = []) => {
          const filtered = old.filter(
            (bookmark) => bookmark.movieId !== movieId
          );
          if (existingBookmark) {
            return [
              ...filtered,
              {
                ...existingBookmark,
                deletedAt: existingBookmark.deletedAt
                  ? null
                  : new Date().toISOString(),
              },
            ];
          }
          const optimisticBookmark: SafeBookmark = {
            id: `temp-${Date.now()}`,
            movieId,
            createdAt: new Date().toISOString(),
            deletedAt: null,
            movie: {
              movieId,
              title: "",
              posterPath: "",
              overview: null,
            },
          };
          return [...filtered, optimisticBookmark];
        }
      );

      if (previousUser) {
        const bookmarkDelta = existingBookmark
          ? existingBookmark.deletedAt
            ? 1
            : -1
          : 1;

        queryClient.setQueryData<SafeCurrentUser>(CURRENT_USER_QUERY_KEY, {
          ...previousUser,
          bookmarkCount: (previousUser.bookmarkCount || 0) + bookmarkDelta,
        });
      }

      return { previousBookmarks, previousUser };
    },
    onError: (_err, _variables, context) => {
      if (context?.previousBookmarks) {
        queryClient.setQueryData(BOOKMARK_QUERY_KEY, context.previousBookmarks);
      }
      if (context?.previousUser) {
        queryClient.setQueryData(CURRENT_USER_QUERY_KEY, context.previousUser);
      }
      toast.error("Failed to update bookmark. Please try again.");
    },
    onSuccess: (response, _variables, context) => {
      const message = response.data.bookmarked
        ? "Bookmark saved successfully"
        : "Bookmark removed successfully";

      toast.message(message, {
        action: {
          label: "Undo",
          onClick: () => {
            if (context?.previousBookmarks) {
              queryClient.setQueryData(
                BOOKMARK_QUERY_KEY,
                context.previousBookmarks
              );
            }
            if (context?.previousUser) {
              queryClient.setQueryData(
                CURRENT_USER_QUERY_KEY,
                context.previousUser
              );
            }
            toggleBookmark.mutate();
          },
        },
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: BOOKMARK_QUERY_KEY });
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
    },
  });

  const isBookmarked =
    bookmarks?.some(
      (bookmark) => bookmark.movieId === movieId && !bookmark.deletedAt
    ) ?? false;

  return {
    isBookmarked,
    isPending: isPending || toggleBookmark.isPending,
    toggleBookmark: () => toggleBookmark.mutate(),
  };
};
