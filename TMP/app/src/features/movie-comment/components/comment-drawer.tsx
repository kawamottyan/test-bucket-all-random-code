"use client";

import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useState } from "react";
import { SafeComment } from "../types";
import CommentFilter from "./comment-filter";
import CommentPost from "./comment-post";
import CommentThreadView from "./comment-thread-view";
import CommentsView from "./comment-view";

interface CommentDrawerProps {
  movieId: number;
  focusedComment: SafeComment | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const CommentDrawer: React.FC<CommentDrawerProps> = ({
  movieId,
  focusedComment,
  open,
  onOpenChange,
}) => {
  const [showAllComments, setShowAllComments] = useState(false);
  const isThreadView = Boolean(focusedComment) && !showAllComments;

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent className="mx-auto h-4/5 max-w-xl">
        <DrawerHeader>
          <div className="flex items-center justify-between">
            <DrawerTitle>
              {isThreadView ? "Comment Thread" : "Comments"}
            </DrawerTitle>
            <DrawerDescription />
            <CommentFilter movieId={movieId} />
          </div>
        </DrawerHeader>
        <ScrollArea>
          {isThreadView && focusedComment ? (
            <CommentThreadView
              movieId={movieId}
              focusedComment={focusedComment}
              onBack={() => setShowAllComments(true)}
            />
          ) : (
            <CommentsView
              movieId={movieId}
              focusedId={focusedComment?.id ?? null}
              open={open}
            />
          )}
        </ScrollArea>
        <DrawerFooter>
          <CommentPost movieId={movieId} />
        </DrawerFooter>
      </DrawerContent>
    </Drawer>
  );
};

export default CommentDrawer;
