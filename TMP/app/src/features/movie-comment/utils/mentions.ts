import { db } from "@/lib/db";
import { MENTION_REGEX } from "../constants";
import { MentionedUser } from "../types";

interface CommentMentionResult {
  processedContent: string;
  mentionedUsers: MentionedUser[];
}

export async function processMentionsInComment(
  content: string
): Promise<CommentMentionResult> {
  const mentions: RegExpMatchArray | null = content.match(MENTION_REGEX);
  if (!mentions) return { processedContent: content, mentionedUsers: [] };

  const uniqueMentionedUsernames = [
    ...new Set(mentions.map((mention) => mention.slice(1))),
  ];

  const uniqueMentionedUsers = await db.user.findMany({
    where: {
      username: { in: uniqueMentionedUsernames },
    },
    select: {
      id: true,
      email: true,
      username: true,
      allowEmailNotification: true,
    },
  });

  const validMentionedUsers = uniqueMentionedUsers.filter(
    (user): user is MentionedUser =>
      user.email !== null && user.username !== null
  );

  let processedContent = content;
  validMentionedUsers.forEach((user) => {
    const mentionRegex = new RegExp(`@${user.username}`, "g");
    processedContent = processedContent.replace(mentionRegex, `@[${user.id}]`);
  });

  return {
    processedContent,
    mentionedUsers: validMentionedUsers,
  };
}

export function formatMentionsInComment(
  content: string,
  mentionMap: Map<string, string>
): string {
  if (!mentionMap.size) return content;

  return content.replace(/@\[([^\]]+)\]/g, (match, userId) => {
    const username = mentionMap.get(userId);
    return username ? `@${username}` : match;
  });
}
