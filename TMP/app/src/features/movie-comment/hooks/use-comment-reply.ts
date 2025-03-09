import { useReplyStore } from "@/stores/use-reply-store";
import { MAX_NESTABLE_DEPTH } from "../constants";
import { SafeComment } from "../types";

export function useCommentReply() {
  const setReply = useReplyStore((state) => state.setReply);

  const handleReply = (comment: SafeComment) => {
    if (!comment.user.username) return;

    const depth = comment.depth;

    const replyToCommentId =
      depth >= MAX_NESTABLE_DEPTH ? comment.parentId : comment.id;
    if (!replyToCommentId) return;

    setReply({
      commentId: replyToCommentId,
      username: comment.user.username,
      content: comment.content,
      depth: depth,
    });
  };

  return { handleReply };
}
