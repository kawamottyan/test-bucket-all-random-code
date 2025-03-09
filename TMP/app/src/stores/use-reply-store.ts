import { create } from "zustand";

interface ReplyContext {
  commentId: string;
  username: string;
  content: string;
  depth: number;
}

interface ReplyState {
  replyContext: ReplyContext | null;
  setReply: (context: ReplyContext) => void;
  clearReply: () => void;
}

export const useReplyStore = create<ReplyState>((set) => ({
  replyContext: null,
  setReply: (context) => set({ replyContext: context }),
  clearReply: () => set({ replyContext: null }),
}));
