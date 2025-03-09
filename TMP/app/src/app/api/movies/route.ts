import redis from "@/db/redis";
import { ITEMS_PER_PAGE, REDIS_RECOMMENDATION_PREFIX, REDIS_TRENDING_PREFIX } from "@/features/movie/constants";
import { movieSelect } from "@/features/movie/query";
import { SafeMovie } from "@/features/movie/types";
import { transformToSafeMovie } from "@/features/movie/utils/transform-to-safe-movie";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import {
  InvokeEndpointCommand,
  SageMakerRuntimeClient,
} from "@aws-sdk/client-sagemaker-runtime";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";

interface RecommendationItem {
  movie_id: number;
  score: number;
}

interface RecommendationResponse {
  success: boolean;
  message: string;
  data: RecommendationItem[];
}

interface TrendingMovieData {
  movieIds: number[];
  createdAt: string;
  updatedAt: string;
}

function logError(message: string, error: unknown): void {
  console.log(`[ERROR] ${message}:`, error instanceof Error
    ? { message: error.message, name: error.name }
    : String(error)
  );
}

export async function GET(
  request: NextRequest
): Promise<NextResponse<ServerResponse>> {
  try {
    const url = new URL(request.url);

    const fetchType = url.searchParams.get("fetchType") || "initial";
    const excludedIds = url.searchParams.get("excludedIds")
      ? JSON.parse(url.searchParams.get("excludedIds")!)
      : [];

    const cookieStore = await cookies();
    const sessionId = cookieStore.get("session_id")?.value;

    let movies: SafeMovie[] = [];

    if (fetchType === "initial" && sessionId) {
      movies = await getRecommendationsFromRedis(sessionId);
    } else if (fetchType === "subsequent" && sessionId) {
      movies = await getPersonalizedRecommendations(sessionId, excludedIds, ITEMS_PER_PAGE);
    }

    if (movies.length < ITEMS_PER_PAGE) {
      try {
        const trendingMovies = await getTrendingMovies(excludedIds);

        if (trendingMovies.length > 0) {
          const existingMovieIds = movies.map(movie => movie.movieId);

          const filteredTrendingMovies = trendingMovies.filter(
            movie => !existingMovieIds.includes(movie.movieId)
          );

          movies = [...movies, ...filteredTrendingMovies];
        }
      } catch (err) {
        logError("Error fetching trending movies", err);
      }
    }

    if (movies.length < ITEMS_PER_PAGE) {
      try {
        const allExcludedIds = [
          ...excludedIds,
          ...movies.map(movie => movie.movieId)
        ];

        const randomMovies = await getRandomMovies(allExcludedIds, ITEMS_PER_PAGE - movies.length);

        if (randomMovies.length > 0) {
          movies = [...movies, ...randomMovies];
        }
      } catch (err) {
        logError("Error fetching random movies", err);
      }
    }

    return NextResponse.json({
      success: true,
      message: "Movies fetched successfully",
      data: movies.slice(0, ITEMS_PER_PAGE)
    });
  } catch (err) {
    logError("Critical error while fetching movies", err);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to fetch movies"
      },
      { status: 500 }
    );
  }
}

async function getPersonalizedRecommendations(
  sessionId: string,
  excludedIds: number[],
  count: number
): Promise<SafeMovie[]> {
  try {
    let recommendationData: RecommendationResponse | null = null;

    if (process.env.AWS_SAGEMAKER_ENDPOINT_NAME) {
      recommendationData = await getSageMakerRecommendations(sessionId, excludedIds, count);
    } else if (process.env.RECOMMENDATION_ENDPOINT) {
      recommendationData = await getCustomRecommendations(sessionId, excludedIds, count);
    } else {
      return [];
    }

    if (!recommendationData || !recommendationData.success || !recommendationData.data) {
      return [];
    }

    const recommendedMovieIds = recommendationData.data.map(item => item.movie_id);

    const moviesFromDb = await db.movie.findMany({
      where: {
        movieId: {
          in: recommendedMovieIds
        }
      },
      select: movieSelect
    });

    const sortedMovies = recommendedMovieIds.map(id =>
      moviesFromDb.find(movie => movie.movieId === id)
    ).filter(Boolean);

    return sortedMovies.map(movie => transformToSafeMovie(movie!));
  } catch (err) {
    logError("Error getting personalized recommendations", err);
    return [];
  }
}

async function getSageMakerRecommendations(
  sessionId: string,
  excludedIds: number[],
  count: number
): Promise<RecommendationResponse | null> {
  try {
    const sageMakerClient = new SageMakerRuntimeClient({
      region: process.env.AWS_REGION,
    });

    const payload = {
      model_name: process.env.RECOMMENDATION_MODEL_NAME,
      session_id: sessionId,
      item_size: count,
      excluded_ids: excludedIds.map(id => id.toString())
    };

    const params = {
      EndpointName: process.env.AWS_SAGEMAKER_ENDPOINT_NAME!,
      ContentType: "application/json",
      Body: JSON.stringify(payload),
    };

    const command = new InvokeEndpointCommand(params);
    const response = await sageMakerClient.send(command);
    return JSON.parse(new TextDecoder("utf-8").decode(response.Body));
  } catch (err) {
    logError("Error invoking SageMaker endpoint", err);
    return null;
  }
}

async function getCustomRecommendations(
  sessionId: string,
  excludedIds: number[],
  count: number
): Promise<RecommendationResponse | null> {
  try {
    const payload = {
      model_name: process.env.RECOMMENDATION_MODEL_NAME,
      session_id: sessionId,
      item_size: count,
      excluded_ids: excludedIds.map(id => id.toString())
    };

    const response = await fetch(process.env.RECOMMENDATION_ENDPOINT!, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Recommendation service responded with status: ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    logError("Error calling recommendation endpoint", err);
    return null;
  }
}

async function getRecommendationsFromRedis(sessionId: string): Promise<SafeMovie[]> {
  try {
    const keys = await redis.keys(`${REDIS_RECOMMENDATION_PREFIX}:${sessionId}:*`);

    if (keys.length === 0) {
      return [];
    }

    keys.sort().reverse();
    const latestKey = keys[0];

    const recommendationData = await redis.hgetall(latestKey);

    if (!recommendationData || !recommendationData.similar_movies) {
      return [];
    }

    const similarMovieIds = JSON.parse(recommendationData.similar_movies as string);

    const moviesFromDb = await db.movie.findMany({
      where: {
        movieId: {
          in: similarMovieIds
        }
      },
      select: movieSelect
    });

    return moviesFromDb.map(movie => transformToSafeMovie(movie));
  } catch (err) {
    logError("Error getting recommendations from Redis", err);
    return [];
  }
}

async function getTrendingMovies(excludedIds: number[]): Promise<SafeMovie[]> {
  try {
    const trendingMovieData = await redis.get(REDIS_TRENDING_PREFIX) as TrendingMovieData | null;

    if (!trendingMovieData || !trendingMovieData.movieIds || !Array.isArray(trendingMovieData.movieIds)) {
      return [];
    }

    const moviesFromDb = await db.movie.findMany({
      where: {
        movieId: {
          in: trendingMovieData.movieIds,
          notIn: excludedIds
        }
      },
      select: movieSelect,
      take: ITEMS_PER_PAGE
    });

    return moviesFromDb.map(movie => transformToSafeMovie(movie));
  } catch (err) {
    logError("Error getting trending movies", err);
    return [];
  }
}

async function getRandomMovies(excludedIds: number[], count: number): Promise<SafeMovie[]> {
  try {
    const safeExcludedIds = Array.isArray(excludedIds) ? excludedIds : [];
    const safeCount = typeof count === 'number' && !isNaN(count) && count > 0 ? count : ITEMS_PER_PAGE;

    const moviesFromDb = await db.movie.findMany({
      where: {
        movieId: {
          notIn: safeExcludedIds
        }
      },
      select: movieSelect,
      orderBy: {
        voteCount: "desc"
      },
      take: safeCount
    });

    return moviesFromDb.map(movie => transformToSafeMovie(movie));
  } catch (err) {
    logError("Error getting random movies", err);
    return [];
  }
}