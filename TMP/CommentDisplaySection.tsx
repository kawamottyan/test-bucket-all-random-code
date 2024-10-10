import React, { useEffect } from "react";
import { Button } from "../ui/button";
import { useInfiniteScroll } from "@/hooks/useInfiniteScroll";
import { SafeComment, SafeUser } from "@/types";
import CommentDisplayCard from "./CommentDisplayCard";

interface CommentDisplaySectionProps {
  comments: SafeComment[];
  fetchMoreComments: () => void;
  hasMore: boolean;
  currentUser: SafeUser | null;
  movieId: number;
  onCommentSubmit: (newComment: SafeComment) => void;
}

const CommentDisplaySection: React.FC<CommentDisplaySectionProps> = ({
  comments,
  fetchMoreComments,
  hasMore,
  currentUser,
  movieId,
  onCommentSubmit,
}) => {
  const { lastElementRef } = useInfiniteScroll(hasMore, fetchMoreComments);

  useEffect(() => {
    fetchMoreComments();
  }, [fetchMoreComments]);

  return (
    <div className="flex flex-col gap-y-6">
      {comments.map((comment, index) => {
        if (comments.length === index + 1) {
          return (
            <div ref={lastElementRef} key={comment.id}>
              <CommentDisplayCard
                comment={comment}
                currentUser={currentUser}
                movieId={movieId}
                onCommentSubmit={onCommentSubmit}
              />
            </div>
          );
        } else {
          return (
            <CommentDisplayCard
              key={comment.id}
              comment={comment}
              currentUser={currentUser}
              movieId={movieId}
              onCommentSubmit={onCommentSubmit}
            />
          );
        }
      })}
      {hasMore && (
        <div className="flex justify-center my-4">
          <Button variant="ghost" disabled>
            Loading...
          </Button>
        </div>
      )}
    </div>
  );
};

export default CommentDisplaySection;
