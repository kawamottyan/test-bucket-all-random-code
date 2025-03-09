import { Button } from "@/components/ui/button";
import { useReplyStore } from "@/stores/use-reply-store";
import { ChevronLeft } from "lucide-react";
import { SafeComment } from "../types";
import CommentDisplayCard from "./comment-display-card";

interface CommentThreadViewProps {
  movieId: number;
  focusedComment: SafeComment;
  onBack: () => void;
}

const CommentThreadView: React.FC<CommentThreadViewProps> = ({
  movieId,
  focusedComment,
  onBack,
}) => {
  const clearReply = useReplyStore((state) => state.clearReply);

  const handleBack = () => {
    clearReply();
    onBack();
  };

  if (!focusedComment) return null;
  return (
    <div className="flex flex-col">
      <Button
        variant="ghost"
        onClick={handleBack}
        className="flex w-fit items-center gap-x-2 text-muted-foreground"
      >
        <ChevronLeft className="h-4 w-4" />
        All Comments
      </Button>
      <CommentDisplayCard
        key={focusedComment.id}
        movieId={movieId}
        comment={focusedComment}
        parentId={null}
        focusedId={focusedComment.id}
        depth={1}
        isThreadView={true}
      />
    </div>
  );
};

export default CommentThreadView;
