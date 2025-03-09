import getCurrentUser from "@/actions/get-user";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
  commentId: string;
}

export async function PATCH(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [{ movieId, commentId }, currentUser] = await Promise.all([
      Promise.resolve(params),
      getCurrentUser(),
    ]);

    const movieIdNumber = parseInt(movieId, 10);

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    if (!movieIdNumber || !commentId) {
      return NextResponse.json(
        {
          success: false,
          message: "Missing required fields",
        },
        { status: 400 }
      );
    }

    const existingFavorite = await db.favoriteComment.findUnique({
      where: {
        userId_commentId: {
          userId: currentUser.id,
          commentId: commentId,
        },
      },
    });

    if (!existingFavorite) {
      await db.$transaction([
        db.favoriteComment.create({
          data: {
            userId: currentUser.id,
            movieId: movieIdNumber,
            commentId: commentId,
            deletedAt: null,
          },
        }),
        db.movieComment.update({
          where: { id: commentId },
          data: { favoriteCount: { increment: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Comment favorited",
          data: {
            favorited: true
          }
        },
        { status: 200 }
      );
    }

    if (existingFavorite.deletedAt) {
      await db.$transaction([
        db.favoriteComment.update({
          where: { id: existingFavorite.id },
          data: { deletedAt: null },
        }),
        db.movieComment.update({
          where: { id: commentId },
          data: { favoriteCount: { increment: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Comment favorited",
          data: {
            favorited: true
          }
        },
        { status: 200 }
      );
    } else {
      await db.$transaction([
        db.favoriteComment.update({
          where: { id: existingFavorite.id },
          data: { deletedAt: new Date() },
        }),
        db.movieComment.update({
          where: { id: commentId },
          data: { favoriteCount: { decrement: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Comment unfavorited",
          data: {
            favorited: false
          }
        },
        { status: 200 }
      );
    }
  } catch (error: unknown) {
    console.error("Failed to toggle favorite:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to toggle favorite. Please try again.",
      },
      { status: 500 }
    );
  }
}
