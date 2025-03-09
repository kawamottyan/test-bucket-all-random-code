import getCurrentUser from "@/actions/get-user";
import { COMMENTS_PER_PAGE } from "@/features/movie-comment/constants";
import { commentInclude } from "@/features/movie-comment/query";
import { CommentApiSchema } from "@/features/movie-comment/schemas";
import { CommentPage, SafeComment } from "@/features/movie-comment/types";
import { formatMentionsInComment, processMentionsInComment } from "@/features/movie-comment/utils/mentions";
import { generateUniqueCommentSlug } from "@/features/movie-comment/utils/slug";
import { transformToSafeComment } from "@/features/movie-comment/utils/transform-to-safe-comment";
import { createNotifications } from "@/features/notification/actions/create-notification";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextRequest, NextResponse } from "next/server";

interface RequestParams {
  movieId: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const { movieId } = await params;
    const movieIdNumber = parseInt(movieId, 10);
    const searchParams = request.nextUrl.searchParams;
    const parentId = searchParams.get('parentId') || null;
    const cursor = searchParams.get('cursor') || undefined;

    if (!movieId || isNaN(movieIdNumber)) {
      return NextResponse.json(
        {
          success: false,
          message: "Invalid movie ID",
        },
        { status: 400 }
      );
    }

    const currentUser = await getCurrentUser();

    const comments = await db.movieComment.findMany({
      where: {
        movieId: movieIdNumber,
        parentId: parentId,
      },
      cursor: cursor ? { id: cursor } : undefined,
      skip: cursor ? 1 : 0,
      take: COMMENTS_PER_PAGE + 1,
      orderBy: {
        createdAt: "desc",
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

    const hasMore = comments.length > COMMENTS_PER_PAGE;
    const currentPageComments = hasMore ? comments.slice(0, -1) : comments;

    const mentionMap = new Map<string, string>();
    currentPageComments.forEach((comment) => {
      comment.mentions
        .filter((mention) => !mention.deletedAt && mention.user?.id)
        .forEach((mention) => {
          mentionMap.set(mention.user.id, mention.user.username ?? "[deleted]");
        });
    });

    const formattedComments = currentPageComments.map((comment) => ({
      ...comment,
      content: formatMentionsInComment(comment.content, mentionMap),
    }));

    const safeComments: SafeComment[] = formattedComments.map((comment) =>
      transformToSafeComment(comment, currentUser?.id)
    );

    const commentPage: CommentPage = {
      comments: safeComments,
      nextCursor: hasMore ? comments[COMMENTS_PER_PAGE].id : undefined,
      hasMore,
    };

    return NextResponse.json(
      {
        success: true,
        message: "Comments fetched successfully",
        data: commentPage,
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Failed to fetch comments:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Unable to fetch comments. Please try again.",
      },
      { status: 500 }
    );
  }
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

    if (!currentUser || !currentUser.username) {
      return NextResponse.json(
        {
          success: false,
          message: "Unauthorized: Please log in",
        },
        { status: 401 }
      );
    }

    const validatedData = CommentApiSchema.safeParse({
      ...body,
      movieId: movieIdNumber,
    });

    if (!validatedData.success) {
      return NextResponse.json(
        {
          success: false,
          message: "Invalid input data",
          errors: validatedData.error.errors,
        },
        { status: 400 }
      );
    }

    const { content, depth, parentId, isSpoiler } = validatedData.data;

    const { processedContent, mentionedUsers } =
      await processMentionsInComment(content);

    const slug = await generateUniqueCommentSlug();

    await db.$transaction(async (tx) => {
      const comment = await tx.movieComment.create({
        data: {
          userId: currentUser.id,
          movieId: movieIdNumber,
          content: processedContent,
          slug,
          depth,
          parentId: parentId || null,
          isSpoiler,
          deletedAt: null,
        },
      });

      if (mentionedUsers.length > 0) {
        await tx.commentMention.createMany({
          data: mentionedUsers.map((user) => ({
            commentId: comment.id,
            userId: user.id,
            movieId: movieIdNumber,
            deletedAt: null,
          })),
        });
      }

      await tx.movie.update({
        where: { movieId: movieIdNumber },
        data: { commentCount: { increment: 1 } },
      });

      if (parentId) {
        await tx.movieComment.update({
          where: { id: parentId },
          data: { replyCount: { increment: 1 } },
        });
      }

      await createNotifications({
        movieId: comment.movieId,
        slug: comment.slug,
        sender: currentUser,
        mentionedUsers,
        parentId: comment.parentId,
      });
    });

    return NextResponse.json(
      {
        success: true,
        message: "Comment created successfully",
      },
      { status: 201 }
    );
  } catch (error: unknown) {
    console.error("Failed to create your comment:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to create your comment. Please try again.",
      },
      { status: 500 }
    );
  }
}
