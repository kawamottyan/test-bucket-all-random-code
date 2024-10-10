import { useState, useEffect } from "react";
import { DrawerContent, DrawerDescription, DrawerFooter, DrawerTitle } from "../ui/drawer";
import { ScrollArea } from "../ui/scroll-area";
import CommentPost from "./CommentPost";
import CommentDisplaySection from "./CommentDisplaySection";
import { SafeComment } from "@/types";
import { useCurrentUserStore } from "@/stores/currentuserStore";
import { useFetchComments } from "@/hooks/useFetchComments";

interface CommentDrawerProps {
  movieId: number;
}

const CommentDrawer: React.FC<CommentDrawerProps> = ({ movieId }) => {
  const currentUser = useCurrentUserStore((state) => state.currentUser);
  const [comments, setComments] = useState<SafeComment[]>([]);

  const { comments: fetchedComments, fetchComments, hasMore } = useFetchComments({ movieId, take: 3 });

  useEffect(() => {
    setComments(fetchedComments);
  }, [fetchedComments]);

  const handleCommentSubmit = (newComment: SafeComment) => {
    setComments((prevComments) => [newComment, ...prevComments]);
  };

  return (
    <DrawerContent className="max-w-xl mx-auto h-4/5">
      <DrawerTitle></DrawerTitle>
      <DrawerDescription></DrawerDescription>
      <ScrollArea className="h-3/5">
        <CommentDisplaySection
          comments={comments}
          fetchMoreComments={fetchComments}
          hasMore={hasMore}
          currentUser={currentUser}
          movieId={movieId}
          onCommentSubmit={handleCommentSubmit}
        />
      </ScrollArea>
      <DrawerFooter>
        <CommentPost
          currentUser={currentUser}
          movieId={movieId}
          onCommentSubmit={handleCommentSubmit}
        />
      </DrawerFooter>
    </DrawerContent>
  );
};

export default CommentDrawer;
