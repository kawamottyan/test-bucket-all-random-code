import { movieSelect } from "@/features/movie/query";
import { transformToSafeMovie } from "@/features/movie/utils/transform-to-safe-movie";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const resolvedParams = await params;
    const movieId = resolvedParams.movieId;
    const movieIdNumber = parseInt(movieId, 10);

    if (!movieIdNumber || isNaN(movieIdNumber)) {
      return NextResponse.json(
        {
          success: false,
          message: "Invalid movie ID",
        },
        { status: 400 }
      );
    }

    const movie = await db.movie.findUnique({
      where: {
        movieId: movieIdNumber,
      },
      select: movieSelect,
    });

    if (!movie) {
      return NextResponse.json(
        {
          success: false,
          message: "Movie not found",
        },
        { status: 404 }
      );
    }

    const safeMovie = transformToSafeMovie(movie);

    return NextResponse.json({
      success: true,
      message: "Movie fetched successfully",
      data: safeMovie,
    });
  } catch (error) {
    console.error("Error fetching movie:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to fetch movie",
      },
      { status: 500 }
    );
  }
}