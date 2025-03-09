import { MOVIE_QUERY_KEY } from "@/constants/query-keys";
import { useInfiniteQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { MovieFetchType, SafeMovie } from "../types";

interface MoviesResponse {
  success: boolean;
  message: string;
  data: SafeMovie[];
}

export const useFetchInfiniteMovies = () => {
  const { data, fetchNextPage, isFetchingNextPage, isLoading, error } = useInfiniteQuery<
    MoviesResponse
  >({
    queryKey: MOVIE_QUERY_KEY,
    queryFn: async ({ pageParam = "initial" as MovieFetchType }) => {
      const currentMovies: SafeMovie[] = data?.pages.flatMap((page: MoviesResponse) => page.data) ?? [];
      const excludedMovieIds: number[] = currentMovies.map(
        (movie) => movie.movieId
      );
      
      let apiUrl = `/api/movies?fetchType=${pageParam}`;
      if (excludedMovieIds.length > 0) {
        apiUrl += `&excludedIds=${JSON.stringify(excludedMovieIds)}`;
      }
      
      const response = await fetch(apiUrl);
      
      if (!response.ok) {
        throw new Error("Failed to fetch movies");
      }
      
      const responseData = await response.json();
      
      if (!responseData.success) {
        throw new Error(responseData.message || "Unknown error");
      }
      
      return responseData;
    },
    getNextPageParam: () => "subsequent" as MovieFetchType,
    initialPageParam: "initial" as MovieFetchType,
  });

  const movies = useMemo(() => 
    data?.pages.flatMap((page: MoviesResponse) => page.data) ?? [], 
    [data?.pages]
  );

  return {
    movies,
    fetchNextPage,
    isFetchingNextPage,
    isLoading,
    error,
  };
};