import { VideoItem } from "@prisma/client";

export type SafeMovie = {
  movieId: number;
  title: string;
  tagline: string | null;
  releaseDate: string | null;
  runtime: string | null;
  overview: string;
  posterPath: string;
  commentCount: number;
  genres: string[];
  videos: SafeVideo[];
};

export interface SafeVideo {
  id: string;
  videoPath: string;
  name: string;
  type: string;
  official: boolean;
  size: number;
}

export type SafeVideoItem = Omit<VideoItem, "publishedAt "> & {
  publishedAt: string | null;
};

export type MovieFetchType = "initial" | "subsequent";