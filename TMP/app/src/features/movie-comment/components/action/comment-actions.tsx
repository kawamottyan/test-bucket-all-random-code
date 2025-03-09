"use client";

import AuthButton from "@/components/auth-button";
import { Button } from "@/components/ui/button";
import { AtSign, Heart, MessageCircle } from "lucide-react";
import React, { useEffect } from "react";
import { MAX_NESTABLE_DEPTH } from "../../constants";
import { useCommentFavoriteActions } from "../../hooks/use-comment-favorite-actions";
import { useCommentReply } from "../../hooks/use-comment-reply";
import { SafeComment } from "../../types";
import CommentActionsMenu from "./comment-action-menu";

interface CommentActionsProps {
  movieId: number;
  comment: SafeComment;
  parentId: string | null;
  isOwner: boolean;
  isFavorite: boolean;
  isThreadView: boolean;
}

const CommentActions: React.FC<CommentActionsProps> = ({
  movieId,
  comment,
  parentId,
  isOwner,
  isFavorite,
  isThreadView = false,
}) => {
  const { mutate: toggleFavorite, isPending } =
    useCommentFavoriteActions(movieId);
  const { handleReply } = useCommentReply();
  const isMaxDepth = comment.depth >= MAX_NESTABLE_DEPTH;

  const handleFavorite = () => {
    toggleFavorite({
      commentId: comment.id,
      isFavorite,
    });
  };

  useEffect(() => {
    if (isThreadView) {
      handleReply(comment);
    }
  }, [isThreadView, comment, parentId, handleReply]);

  const replyIcon = isMaxDepth ? (
    <AtSign className="h-4 w-4" />
  ) : (
    <MessageCircle className="h-4 w-4" />
  );

  return (
    <div className="ml-auto flex">
      <AuthButton
        variant="ghost"
        size="icon"
        onClick={handleFavorite}
        icon={
          <Heart className={`h-4 w-4 ${isFavorite ? "fill-primary" : ""}`} />
        }
        className="hover:bg-secondary"
        disabled={isPending}
      />
      <Button variant="ghost" size="icon" onClick={() => handleReply(comment)}>
        {replyIcon}
      </Button>
      <CommentActionsMenu
        movieId={movieId}
        commentId={comment.id}
        content={comment.content}
        isOwner={isOwner}
        isSpoiler={comment.isSpoiler}
      />
    </div>
  );
};

export default CommentActions;
