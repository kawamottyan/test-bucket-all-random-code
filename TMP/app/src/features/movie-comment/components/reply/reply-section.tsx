"use client";

import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useFetchComments } from "@/features/movie-comment/hooks/use-fetch-comments";
import { SafeComment } from "@/features/movie-comment/types";
import { ChevronDown, ChevronUp, CornerDownRight } from "lucide-react";
import React, { useState } from "react";
import CommentDisplayCard from "../comment-display-card";
import CommentSkeleton from "../comment-skeleton";

interface ReplySectionProps {
  movieId: number;
  comment: SafeComment;
  focusedId: string | null;
  depth: number;
}

const ReplySection: React.FC<ReplySectionProps> = ({
  movieId,
  comment,
  focusedId,
  depth,
}) => {
  const [isRepliesOpen, setIsRepliesOpen] = useState(false);

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } =
    useFetchComments({ movieId, parentId: comment.id, enabled: isRepliesOpen });

  const replies = data?.pages.flatMap((page) => page.comments) ?? [];

  const handleLoadMore = () => {
    if (!isFetchingNextPage) {
      fetchNextPage();
    }
  };

  return (
    <Collapsible open={isRepliesOpen} onOpenChange={setIsRepliesOpen}>
      <CollapsibleTrigger asChild>
        <Button variant="ghost" size="sm" className="text-muted-foreground">
          {isRepliesOpen ? (
            <ChevronUp className="mr-1 h-4 w-4" />
          ) : (
            <CornerDownRight className="mr-1 h-4 w-4" />
          )}
          {isRepliesOpen
            ? `Hide replies`
            : `Show ${comment.replyCount} replies`}
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="flex flex-col pb-4">
          {isPending ? (
            <CommentSkeleton />
          ) : (
            replies.map((reply) => (
              <CommentDisplayCard
                key={reply.id}
                movieId={movieId}
                comment={reply}
                parentId={comment.id}
                focusedId={focusedId}
                depth={depth + 1}
              />
            ))
          )}
          {hasNextPage && !isFetchingNextPage && (
            <Button
              variant="ghost"
              className="self-start text-muted-foreground"
              onClick={handleLoadMore}
            >
              <ChevronDown className="mr-1 h-4 w-4" />
              Show more replies
            </Button>
          )}
          {isFetchingNextPage && <CommentSkeleton />}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default ReplySection;
