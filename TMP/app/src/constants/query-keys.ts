export const CURRENT_USER_QUERY_KEY = ["user"] as const;
export const MOVIE_ID_QUERY_KEY = "data-movie-id";
export const MOVIE_QUERY_KEY = ["movies"] as const;
export const NOTIFICATION_QUERY_KEY = ["notifications"] as const;
export const BOOKMARK_QUERY_KEY = ["movie-bookmarks"] as const;
export const REVIEW_QUERY_KEY = ["movie-reviews"] as const;
export const COMMENTS_QUERY_KEY = (movieId: number) => ["comments", movieId];
