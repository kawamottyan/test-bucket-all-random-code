import { MovieComment } from "@prisma/client";

export type SafeComment = Omit<
  MovieComment,
  | "editedAt"
  | "deletedAt"
  | "createdAt"
  | "updatedAt"
  | "latestActivityAt"
  | "userId"
> & {
  editedAt: string | null;
  deletedAt: string | null;
  createdAt: string;
  latestActivityAt: string;
  user: {
    name: string;
    username: string;
    image: string | null;
  };
  mentions: {
    username: string;
    image: string | null;
  }[];
  isOwner: boolean;
  isFavorite: boolean;
};

// export type ReplyType = 'NESTED' | 'MENTION';

// export type ReplyContext = {
//     commentId: string | undefined;
//     username: string | undefined;
//     content: string | undefined;
//     depth: number;
//     replyType: ReplyType;
// };

export type MentionedUser = {
  id: string;
  email: string;
  username: string;
  allowEmailNotification: boolean;
};

export type CommentPage = {
  comments: SafeComment[];
  nextCursor?: string;
  hasMore: boolean;
};

export type CommentType = "mention" | "reply";
