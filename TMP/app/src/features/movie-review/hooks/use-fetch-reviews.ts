import { REVIEW_QUERY_KEY } from "@/constants/query-keys";
import { ServerResponse } from "@/types";
import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { SafeReview } from "../types";

interface ReviewsResponse extends ServerResponse {
  data?: SafeReview[];
}

export const useFetchReviews = () => {
  const { data: session } = useSession();

  return useQuery<SafeReview[]>({
    queryKey: REVIEW_QUERY_KEY,
    queryFn: async () => {
      const response = await fetch('/api/user/reviews');

      if (!response.ok) {
        throw new Error(`Failed to fetch reviews: ${response.status}`);
      }

      const result: ReviewsResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      return result.data || [];
    },
    enabled: !!session,
  });
};