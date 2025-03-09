import getCurrentUser from "@/actions/get-user";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
}

export async function POST(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [{ movieId }, currentUser] = await Promise.all([
      Promise.resolve(params),
      getCurrentUser(),
    ]);

    const movieIdNumber = parseInt(movieId, 10);

    if (!currentUser) {
      return NextResponse.json(
        {
          success: false,
          message: "Unauthorized: Please log in",
        },
        { status: 401 }
      );
    }

    if (!movieIdNumber) {
      return NextResponse.json(
        {
          success: false,
          message: "Missing required fields",
        },
        { status: 400 }
      );
    }

    const existingBookmark = await db.movieBookmark.findUnique({
      where: {
        userId_movieId: {
          userId: currentUser.id,
          movieId: movieIdNumber,
        },
      },
    });

    if (!existingBookmark) {
      await db.$transaction([
        db.movieBookmark.create({
          data: {
            userId: currentUser.id,
            movieId: movieIdNumber,
            deletedAt: null,
          },
        }),
        db.user.update({
          where: { id: currentUser.id },
          data: { bookmarkCount: { increment: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Bookmark created successfully",
          data: { bookmarked: true },
        },
        { status: 200 }
      );
    }

    if (existingBookmark.deletedAt) {
      await db.$transaction([
        db.movieBookmark.update({
          where: { id: existingBookmark.id },
          data: { deletedAt: null },
        }),
        db.user.update({
          where: { id: currentUser.id },
          data: { bookmarkCount: { increment: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Bookmark restored successfully",
          data: { bookmarked: true },
        },
        { status: 200 }
      );
    } else {
      await db.$transaction([
        db.movieBookmark.update({
          where: { id: existingBookmark.id },
          data: { deletedAt: new Date() },
        }),
        db.user.update({
          where: { id: currentUser.id },
          data: { bookmarkCount: { decrement: 1 } },
        }),
      ]);

      return NextResponse.json(
        {
          success: true,
          message: "Bookmark removed successfully",
          data: { bookmarked: false },
        },
        { status: 200 }
      );
    }
  } catch (error: unknown) {
    console.error("Failed to toggle bookmark:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to toggle bookmark. Please try again.",
      },
      { status: 500 }
    );
  }
}
