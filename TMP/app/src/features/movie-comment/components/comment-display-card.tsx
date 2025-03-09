"use client";

import React from "react";
import { SafeComment } from "../types";
import CommentActions from "./action/comment-actions";
import CommentContent from "./comment-content";
import CommentUserSection from "./comment-user-section";
import ReplySection from "./reply/reply-section";

interface CommentDisplayCardProps {
  movieId: number;
  comment: SafeComment;
  parentId: string | null;
  focusedId: string | null;
  depth: number;
  isThreadView?: boolean;
}

const CommentDisplayCard: React.FC<CommentDisplayCardProps> = ({
  movieId,
  comment,
  parentId,
  focusedId,
  depth,
  isThreadView = false,
}) => {
  return (
    <div
      className={`${depth === 1 ? "" : "border-l border-muted-foreground"} pl-4 pt-4`}
    >
      <CommentUserSection
        user={comment.user}
        createdAt={comment.createdAt}
        depth={depth}
      />
      <div className="flex items-center">
        <div className="flex-1">
          <CommentContent
            movieId={movieId}
            content={comment.content}
            isSpoiler={comment.isSpoiler}
            isFocused={focusedId === comment.id}
            isDeleted={comment.deletedAt !== null}
          />
        </div>
        <div
          className={`shrink-0 ${comment.deletedAt ? "invisible" : ""}`}
        >
          <CommentActions
            movieId={movieId}
            comment={comment}
            parentId={parentId}
            isOwner={comment.isOwner}
            isFavorite={comment.isFavorite}
            isThreadView={isThreadView}
          />
        </div>
      </div>
      {comment.replyCount > 0 && (
        <ReplySection
          movieId={movieId}
          comment={comment}
          focusedId={focusedId}
          depth={depth}
        />
      )}
    </div>
  );
};

export default CommentDisplayCard;
