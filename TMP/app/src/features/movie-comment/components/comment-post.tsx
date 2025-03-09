"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useReplyStore } from "@/stores/use-reply-store";
import { Send } from "lucide-react";
import { useSession } from "next-auth/react";
import { useState } from "react";
import { toast } from "sonner";
import { MAX_NESTABLE_DEPTH } from "../constants";
import { useCommentActions } from "../hooks/use-comment-actions";
import { CommentPostForm } from "./comment-post-form";
import ReplyContextCard from "./reply/reply-context-card";

interface CommentPostProps {
  movieId: number;
}

const PLACEHOLDER_TEXT = "Type your comment here...";
const LOGIN_PROMPT_TEXT = "Log in to post a comment";

const CommentPost: React.FC<CommentPostProps> = ({ movieId }) => {
  const { data: session } = useSession();
  const [input, setInput] = useState("");
  const [isSpoiler, setIsSpoiler] = useState(false);
  const [isConfirmModalOpen, setConfirmModalOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  const replyContext = useReplyStore((state) => state.replyContext);
  const clearReply = useReplyStore((state) => state.clearReply);
  const { createComment, isPending } = useCommentActions(movieId);

  const handleSubmit = async () => {
    try {
      const isMaxDepth =
        replyContext && replyContext.depth >= MAX_NESTABLE_DEPTH;

      const getCommentDepth = () => {
        if (!replyContext) return 1;

        if (isMaxDepth) {
          return MAX_NESTABLE_DEPTH;
        }
        return replyContext.depth + 1;
      };

      const content = isMaxDepth
        ? `@${replyContext.username} ${input}`.trim()
        : input;

      await createComment({
        content,
        depth: getCommentDepth(),
        parentId: replyContext?.commentId ?? null,
        isSpoiler,
      });

      setInput("");
      setIsEditing(false);
      setIsSpoiler(false);
      setConfirmModalOpen(false);
      clearReply();
    } catch {
      toast.error("Failed to post comment");
    }
  };

  return (
    <div className="flex flex-col space-y-4">
      <ReplyContextCard />
      <div className="flex items-center space-x-2">
        <Input
          id="message"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={session ? PLACEHOLDER_TEXT : LOGIN_PROMPT_TEXT}
          className="text-md grow border-2"
          disabled={!session}
        />
        <Button
          size="icon"
          className="ml-2 h-10 w-12"
          disabled={!session || isPending || !input.trim()}
          onClick={() => setConfirmModalOpen(true)}
        >
          <Send className="h-5 w-5" />
        </Button>
        <CommentPostForm
          defaultValues={{
            content: input,
            isSpoiler: isSpoiler,
          }}
          setInput={setInput}
          isEditing={isEditing}
          setIsEditing={setIsEditing}
          open={isConfirmModalOpen}
          onOpenChange={setConfirmModalOpen}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  );
};

export default CommentPost;
