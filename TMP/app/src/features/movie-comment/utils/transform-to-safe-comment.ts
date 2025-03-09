import { CommentMention, MovieComment } from "@prisma/client";
import { SafeComment } from "../types";

export interface CommentWithRelations extends MovieComment {
  user: {
    name: string | null;
    image: string | null;
    username: string | null;
  };
  mentions: (CommentMention & {
    user: {
      id: string;
      username: string | null;
      image: string | null;
    };
  })[];
  favorites?: {
    id: string;
  }[];
}

export function transformToSafeComment(
  comment: CommentWithRelations,
  currentUserId?: string
): SafeComment {
  const isDeleted = !!comment.deletedAt;
  return {
    id: comment.id,
    movieId: comment.movieId,
    parentId: comment.parentId,
    slug: comment.slug,
    content: isDeleted ? "[deleted]" : comment.content,
    depth: comment.depth,
    favoriteCount: comment.favoriteCount,
    replyCount: comment.replyCount,
    isSpoiler: comment.isSpoiler,
    isDiscussion: comment.isDiscussion,
    reportedStatus: comment.reportedStatus,
    editedAt: comment.editedAt?.toISOString() ?? null,
    deletedAt: comment.deletedAt?.toISOString() ?? null,
    createdAt: comment.createdAt.toISOString(),
    latestActivityAt: comment.latestActivityAt.toISOString(),
    user: {
      name: comment.user.name ?? "[unknown]",
      username: comment.user.username ?? "[unknown]",
      image: comment.user.image ?? null,
    },
    mentions: comment.mentions
      .filter((mention) => !mention.deletedAt)
      .map((mention) => ({
        username: mention.user.username ?? "[deleted]",
        image: mention.user.image ?? null,
      })),
    isOwner: currentUserId ? currentUserId === comment.userId : false,
    isFavorite: comment.favorites ? comment.favorites.length > 0 : false,
  };
}
