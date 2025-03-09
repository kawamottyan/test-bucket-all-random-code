import { BOOKMARK_QUERY_KEY } from "@/constants/query-keys";
import { ServerResponse } from "@/types";
import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { SafeBookmark } from "../types";

interface BookmarksResponse extends ServerResponse {
  data?: SafeBookmark[];
}

export const useFetchBookmarks = () => {
  const { data: session } = useSession();

  return useQuery<SafeBookmark[]>({
    queryKey: BOOKMARK_QUERY_KEY,
    queryFn: async () => {
      const response = await fetch('/api/user/bookmarks');

      if (!response.ok) {
        throw new Error(`Failed to fetch bookmarks: ${response.status}`);
      }

      const result: BookmarksResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      return result.data || [];
    },
    enabled: !!session,
  });
};