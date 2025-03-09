import { MAX_RETRY_COUNT, USERNAME_TOKEN_LENGTH } from "@/constants/index";
import { db } from "@/lib/db";
import { generateToken } from "@/lib/tokens";

export async function generateUniqueUsername(): Promise<string> {
  let retryCount = 0;

  while (retryCount < MAX_RETRY_COUNT) {
    const username = await generateToken({
      length: USERNAME_TOKEN_LENGTH,
      type: "base32",
    });

    const existingUser = await db.user.findUnique({
      where: { username: username },
    });

    if (!existingUser) {
      return username;
    }

    retryCount++;
  }

  throw new Error("Failed to generate unique username after maximum retries");
}
