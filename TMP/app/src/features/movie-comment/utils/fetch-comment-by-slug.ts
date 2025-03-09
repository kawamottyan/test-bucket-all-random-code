import { ServerResponse } from "@/types";
import { CommentType, SafeComment } from "../types";

interface CommentResponse extends ServerResponse {
  data?: SafeComment;
}

export async function fetchCommentBySlug(movieId: number, slug: string, type: CommentType): Promise<SafeComment | null> {
  try {
    const params = new URLSearchParams();
    params.append('type', type);

    const response = await fetch(`/api/movies/${movieId}/comments/${slug}?${params.toString()}`);

    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      throw new Error(`Failed to fetch comment: ${response.status}`);
    }

    const result: CommentResponse = await response.json();

    if (!result.success) {
      throw new Error(result.message);
    }

    return result.data || null;
  } catch (error) {
    console.error("Failed to fetch comment:", error);
    return null;
  }
}