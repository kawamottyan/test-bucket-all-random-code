import getCurrentUser from "@/actions/get-user";
import { CommentReportApiSchema } from "@/features/movie-comment/schemas";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
  commentId: string;
}

export async function POST(
  request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [body, { movieId, commentId }, currentUser] = await Promise.all([
      request.json(),
      Promise.resolve(params),
      getCurrentUser(),
    ]);

    const movieIdNumber = parseInt(movieId, 10);

    const validatedData = CommentReportApiSchema.safeParse({
      ...body,
      movieId: movieIdNumber,
      commentId,
    });

    if (!validatedData.success) {
      return NextResponse.json(
        { success: false, message: "Invalid input data" },
        { status: 400 }
      );
    }

    const { type, description } = validatedData.data;

    await db.commentReport.create({
      data: {
        movieId: movieIdNumber,
        commentId,
        type,
        description,
        userId: currentUser?.id,
      },
    });

    return NextResponse.json(
      { success: true, message: "Report created successfully" },
      { status: 201 }
    );
  } catch (error) {
    console.error("Failed to report comment:", error);
    return NextResponse.json(
      { success: false, message: "Failed to report comment. Please try again" },
      { status: 500 }
    );
  }
}
