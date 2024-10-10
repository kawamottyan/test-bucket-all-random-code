"use client";

import { useState } from "react";
import { SafeUser } from "@/types";
import { Send, Loader2 } from "lucide-react";
import { useSubmitComment } from "@/hooks/useSubmitComment";
import { LoginAccountModal } from "../modal/LoginAccountModal";
import { Button } from "../ui/button";
import { Dialog, DialogTrigger } from "../ui/dialog";
import { Input } from "../ui/input";
import { toast } from "sonner";
import { SafeComment } from "@/types";

interface CommentPostProps {
  currentUser: SafeUser | null;
  movieId: number;
  parentId?: string;
  onCommentSubmit: (newComment: SafeComment) => void;
}

const CommentPost: React.FC<CommentPostProps> = ({ currentUser, movieId, parentId, onCommentSubmit }) => {
  const [input, setInput] = useState("");
  const { isSubmitting, submitComment } = useSubmitComment({ movieId });

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const newComment = await submitComment(input, parentId);
      setInput("");
      toast.success("Your comment has been submitted.");
      onCommentSubmit(newComment);
    } catch {
      toast.error("An error occurred while submitting your comment.");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center">
      <Input
        id="message"
        placeholder={currentUser ? "Type your comment here..." : "Log in to post a comment"}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={isSubmitting || !currentUser}
        className="flex-grow text-md border-2"
      />
      {currentUser ? (
        <Button
          type="submit"
          size="icon"
          className="ml-2 w-12 h-10"
          disabled={isSubmitting || input.trim() === ""}
        >
          {isSubmitting ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Send className="h-5 w-5" />
          )}
        </Button>
      ) : (
        <Dialog>
          <DialogTrigger asChild>
            <Button type="button" size="icon" className="ml-2 w-12 h-10">
              <Send className="h-5 w-5" />
            </Button>
          </DialogTrigger>
          <LoginAccountModal />
        </Dialog>
      )}
    </form>
  );
};

export default CommentPost;
