import getCurrentUser from "@/actions/get-user";
import { ReviewApiSchema } from "@/features/movie-review/schemas";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
}

export async function POST(
  request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [body, { movieId }, currentUser] = await Promise.all([
      request.json(),
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

    if (!movieIdNumber) {
      return NextResponse.json(
        {
          success: false,
          message: "Missing required fields",
        },
        { status: 400 }
      );
    }

    const validatedData = ReviewApiSchema.safeParse({
      ...body,
      movieId: movieIdNumber,
    });

    if (!validatedData.success) {
      return NextResponse.json(
        { success: false, message: "Invalid input data" },
        { status: 400 }
      );
    }

    const { rating, note, watchDate } = validatedData.data;

    const existingReview = await db.movieReview.findUnique({
      where: {
        userId_movieId: {
          userId: currentUser.id,
          movieId: movieIdNumber,
        },
      },
    });

    if (existingReview && !existingReview.deletedAt) {
      await db.movieReview.update({
        where: { id: existingReview.id },
        data: {
          rating,
          note,
          watchedAt: new Date(watchDate),
        },
      });

      return NextResponse.json({
        success: true,
        message: "Review updated successfully",
      });
    }

    if (existingReview?.deletedAt) {
      await db.$transaction([
        db.movieReview.update({
          where: { id: existingReview.id },
          data: {
            deletedAt: null,
            rating,
            note,
            watchedAt: new Date(watchDate),
          },
        }),
        db.user.update({
          where: { id: currentUser.id },
          data: { reviewCount: { increment: 1 } },
        }),
      ]);

      return NextResponse.json({
        success: true,
        message: "Review restored successfully",
      });
    }

    await db.$transaction([
      db.movieReview.create({
        data: {
          userId: currentUser.id,
          movieId: movieIdNumber,
          rating,
          note,
          watchedAt: new Date(watchDate),
          deletedAt: null,
        },
      }),
      db.user.update({
        where: { id: currentUser.id },
        data: { reviewCount: { increment: 1 } },
      }),
    ]);

    return NextResponse.json({
      success: true,
      message: "Review created successfully",
    });
  } catch (error) {
    console.error("Failed to process review:", error);
    return NextResponse.json(
      { success: false, message: "Failed to process review" },
      { status: 500 }
    );
  }
}

export async function DELETE(
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
        { success: false, message: "Unauthorized: Please log in" },
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

    const existingReview = await db.movieReview.findUnique({
      where: {
        userId_movieId: {
          userId: currentUser.id,
          movieId: movieIdNumber,
        },
      },
    });

    if (!existingReview || existingReview.deletedAt) {
      return NextResponse.json(
        { success: false, message: "Review not found" },
        { status: 404 }
      );
    }

    await db.$transaction([
      db.movieReview.update({
        where: { id: existingReview.id },
        data: { deletedAt: new Date() },
      }),
      db.user.update({
        where: { id: currentUser.id },
        data: { reviewCount: { decrement: 1 } },
      }),
    ]);

    return NextResponse.json({
      success: true,
      message: "Review deleted successfully",
      data: { reviewed: false },
    });
  } catch (error) {
    console.error("Failed to delete review:", error);
    return NextResponse.json(
      { success: false, message: "Failed to delete review" },
      { status: 500 }
    );
  }
}
