import { SafeMovie, SafeVideo } from "@/features/movie/types";
import { POSTER_BASE_URL, YOUTUBE_BASE_URL } from "../constants";

interface MovieWithRelations {
  movieId: number;
  title: string;
  tagline: string | null;
  releaseDate: Date | null;
  runtime: number;
  overview: string;
  posterPath: string;
  commentCount: number;
  genres: { id: number; name: string }[];
  videos: {
    videos: {
      videoId: string;
      iso_639_1: string | null;
      iso_3166_1: string | null;
      name: string;
      key: string;
      publishedAt: Date | null;
      site: string;
      size: number;
      type: string;
      official: boolean;
    }[];
  } | null;
}

export function transformToSafeMovie(movie: MovieWithRelations): SafeMovie {
  let videos: SafeVideo[] = [];

  if (movie.videos && Array.isArray(movie.videos.videos)) {
    const youtubeVideos = movie.videos.videos
      .filter(
        (video) =>
          video.site === "YouTube" &&
          video.iso_639_1 === "en" &&
          video.publishedAt != null
      )
      .sort((a, b) => {
        const dateA = new Date(a.publishedAt!);
        const dateB = new Date(b.publishedAt!);
        return dateB.getTime() - dateA.getTime();
      });

    videos = youtubeVideos.map((video) => ({
      id: video.videoId,
      videoPath: `${YOUTUBE_BASE_URL}${video.key}?autoplay=1&mute=1`,
      name: video.name,
      type: video.type,
      official: video.official,
      size: video.size,
    }));
  }

  return {
    movieId: movie.movieId,
    title: movie.title,
    tagline: movie.tagline,
    releaseDate: movie.releaseDate
      ? new Date(movie.releaseDate).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
      : null,
    runtime: movie.runtime ? movie.runtime.toString() : null,
    overview: movie.overview,
    posterPath: `${POSTER_BASE_URL}${movie.posterPath}`,
    commentCount: movie.commentCount,
    genres: movie.genres.map((genre) => genre.name),
    videos: videos,
  };
}
