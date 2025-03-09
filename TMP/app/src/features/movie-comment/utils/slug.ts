import { MAX_RETRY_COUNT } from "@/constants";
import { db } from "@/lib/db";
import { generateToken } from "@/lib/tokens";
import { COMMENT_SLUG_LENGTH } from "../constants";

export async function generateUniqueCommentSlug(): Promise<string> {
  let retryCount = 0;

  while (retryCount < MAX_RETRY_COUNT) {
    const slug = await generateToken({
      length: COMMENT_SLUG_LENGTH,
      type: "urlsafe",
    });

    const existingComment = await db.movieComment.findUnique({
      where: { slug },
    });

    if (!existingComment) {
      return slug;
    }

    retryCount++;
  }

  throw new Error("Failed to generate unique comment ID after maximum retries");
}
