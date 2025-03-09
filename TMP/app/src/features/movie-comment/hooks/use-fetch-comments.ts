import { COMMENTS_QUERY_KEY } from "@/constants/query-keys";
import { ServerResponse } from "@/types";
import { useInfiniteQuery } from "@tanstack/react-query";
import { CommentPage } from "../types";

interface CommentsResponse extends ServerResponse {
  data?: CommentPage;
}

interface FetchCommentsParams {
  movieId: number;
  parentId: string | null;
  enabled: boolean;
}

export const useFetchComments = ({
  movieId,
  parentId = null,
  enabled = true,
}: FetchCommentsParams) => {
  return useInfiniteQuery<CommentPage>({
    queryKey: [...COMMENTS_QUERY_KEY(movieId), parentId],
    queryFn: async ({ pageParam }) => {
      const params = new URLSearchParams();
      if (parentId) params.append('parentId', parentId);
      if (pageParam) params.append('cursor', pageParam as string);

      const response = await fetch(`/api/movies/${movieId}/comments?${params.toString()}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch comments: ${response.status}`);
      }

      const result: CommentsResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      return result.data || { comments: [], hasMore: false, nextCursor: undefined };
    },
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    enabled,
  });
};