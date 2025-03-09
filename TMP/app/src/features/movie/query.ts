export const movieSelect = {
  movieId: true,
  title: true,
  tagline: true,
  releaseDate: true,
  runtime: true,
  posterPath: true,
  overview: true,
  commentCount: true,
  genres: true,
  videos: {
    select: {
      videos: true,
    },
  },
} as const;
