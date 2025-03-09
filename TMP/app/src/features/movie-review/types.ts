export type SafeReview = {
  id: string;
  movieId: number;
  rating: number;
  note: string | null;
  watchedAt: string;
  deletedAt: string | null;
  movie: {
    movieId: number;
    title: string;
    releaseDate: string | null;
  };
};
