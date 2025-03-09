import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { SafeCurrentUser } from "@/types";
import { useQuery } from "@tanstack/react-query";

export const useFetchCurrentUser = () => {
  return useQuery<SafeCurrentUser | null>({
    queryKey: CURRENT_USER_QUERY_KEY,
    queryFn: async () => {
      const response = await fetch('/api/user');
      if (!response.ok) {
        throw new Error('Failed to fetch user');
      }
      return response.json();
    }
  });
};