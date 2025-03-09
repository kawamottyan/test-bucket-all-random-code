export type SafeBookmark = {
  id: string;
  movieId: number;
  createdAt: string;
  deletedAt: string | null;
  movie: {
    movieId: number;
    title: string;
    posterPath: string;
    overview: string | null;
  };
};

export type BookmarkSortOption =
  | "date-desc"
  | "date-asc"
  | "title-asc"
  | "title-desc";
