import getCurrentUser from "@/actions/get-user";
import { commentInclude } from "@/features/movie-comment/query";
import { CommentEditApiSchema } from "@/features/movie-comment/schemas";
import { CommentType } from "@/features/movie-comment/types";
import { processMentionsInComment } from "@/features/movie-comment/utils/mentions";
import { transformToSafeComment } from "@/features/movie-comment/utils/transform-to-safe-comment";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextRequest, NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
  commentId: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const { movieId, commentId } = await params;
    const movieIdNumber = parseInt(movieId, 10);
    const searchParams = request.nextUrl.searchParams;
    const type = searchParams.get('type') as CommentType || 'mention';

    if (!movieIdNumber || isNaN(movieIdNumber) || !commentId) {
      return NextResponse.json(
        {
          success: false,
          message: "Invalid movieId or slug",
        },
        { status: 400 }
      );
    }

    const currentUser = await getCurrentUser();

    const comment = await db.movieComment.findFirst({
      where: {
        slug: commentId,
        movieId: movieIdNumber,
      },
      include: {
        ...commentInclude,
        favorites: currentUser
          ? {
            where: {
              userId: currentUser.id,
              deletedAt: null,
            },
            select: {
              id: true,
            },
          }
          : false,
      },
    });

    if (!comment) {
      return NextResponse.json(
        {
          success: false,
          message: "Comment not found",
        },
        { status: 404 }
      );
    }

    let targetComment = comment;
    if (type === "reply" && comment.parentId) {
      const parentComment = await db.movieComment.findFirst({
        where: {
          id: comment.parentId,
          movieId: movieIdNumber,
        },
        include: {
          ...commentInclude,
          favorites: currentUser
            ? {
              where: {
                userId: currentUser.id,
                deletedAt: null,
              },
              select: {
                id: true,
              },
            }
            : false,
        },
      });

      if (!parentComment) {
        return NextResponse.json(
          {
            success: false,
            message: "Parent comment not found",
          },
          { status: 404 }
        );
      }
      targetComment = parentComment;
    }

    const safeComment = transformToSafeComment(targetComment, currentUser?.id);

    return NextResponse.json(
      {
        success: true,
        message: "Comment fetched successfully",
        data: safeComment,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Failed to fetch comment:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to fetch comment. Please try again.",
      },
      { status: 500 }
    );
  }
}

export async function PATCH(
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

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    const validatedData = CommentEditApiSchema.safeParse({
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

    const { content, isSpoiler } = validatedData.data;

    const existingComment = await db.movieComment.findUnique({
      where: {
        id: commentId,
        userId: currentUser.id,
      },
      include: {
        mentions: {
          where: {
            deletedAt: null,
          },
          select: {
            userId: true,
          },
        },
      },
    });

    if (!existingComment) {
      return NextResponse.json(
        { success: false, message: "Comment not found" },
        { status: 404 }
      );
    }

    if (existingComment.deletedAt) {
      return NextResponse.json(
        { success: false, message: "Comment already deleted" },
        { status: 404 }
      );
    }

    const { processedContent, mentionedUsers } =
      await processMentionsInComment(content);

    const existingMentionUserIds = new Set(
      existingComment.mentions.map((mention) => mention.userId)
    );

    const newMentionUserIds = new Set(mentionedUsers.map((user) => user.id));

    const mentionsToDelete = [...existingMentionUserIds].filter(
      (id) => !newMentionUserIds.has(id)
    );

    const mentionsToAdd = [...newMentionUserIds].filter(
      (id) => !existingMentionUserIds.has(id)
    );

    await db.$transaction(async (tx) => {
      const updatedComment = await tx.movieComment.update({
        where: {
          id: existingComment.id,
        },
        data: {
          content: processedContent,
          isSpoiler: isSpoiler,
        },
      });

      if (mentionsToDelete.length > 0) {
        await tx.commentMention.updateMany({
          where: {
            commentId,
            userId: {
              in: mentionsToDelete,
            },
            deletedAt: null,
          },
          data: {
            deletedAt: new Date(),
          },
        });
      }

      if (mentionsToAdd.length > 0) {
        await tx.commentMention.createMany({
          data: mentionsToAdd.map((userId) => ({
            commentId: updatedComment.id,
            userId: userId,
            movieId: movieIdNumber,
            deletedAt: null,
          })),
        });
      }
    });

    return NextResponse.json({
      success: true,
      message: "Comment updated successfully",
      data: { reviewed: false },
    });
  } catch (error) {
    console.error("Failed to update comment:", error);
    return NextResponse.json(
      { success: false, message: "Failed to update comment" },
      { status: 500 }
    );
  }
}

export async function DELETE(
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

    const existingComment = await db.movieComment.findUnique({
      where: {
        id: commentId,
        userId: currentUser.id,
      },
    });

    if (!existingComment || existingComment.deletedAt) {
      return NextResponse.json(
        { success: false, message: "Comment not found" },
        { status: 404 }
      );
    }

    await db.$transaction([
      db.movieComment.update({
        where: { id: existingComment.id },
        data: { deletedAt: new Date() },
      }),
      db.movie.update({
        where: { movieId: existingComment.movieId },
        data: { commentCount: { decrement: 1 } },
      }),
    ]);

    return NextResponse.json({
      success: true,
      message: "Comment deleted successfully",
      data: { reviewed: false },
    });
  } catch (error) {
    console.error("Failed to delete comment:", error);
    return NextResponse.json(
      { success: false, message: "Failed to delete comment" },
      { status: 500 }
    );
  }
}
