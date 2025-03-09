import { MentionedUser } from "@/features/movie-comment/types";
import { db } from "@/lib/db";
import { sendNotificationEmail } from "@/lib/email";
import { User } from "@prisma/client";

interface CreateNotificationsProps {
  movieId: number;
  slug: string;
  sender: User;
  mentionedUsers: MentionedUser[];
  parentId: string | null;
}

type NotificationPayload = {
  userId: string;
  message: string;
  type: "MENTION" | "REPLY";
  link: string;
  senderId: string;
  readAt: null;
  achievedAt: null;
  deletedAt: null;
};

export const createNotifications = async ({
  movieId,
  slug,
  sender,
  mentionedUsers,
  parentId,
}: CreateNotificationsProps): Promise<void> => {
  try {
    if (mentionedUsers.length === 0 && !parentId) {
      return;
    }

    const notifications: NotificationPayload[] = [];
    const emailNotifications = [];
    const userIds = new Set<string>();

    if (mentionedUsers.length > 0) {
      for (const user of mentionedUsers) {
        notifications.push({
          userId: user.id,
          message: `@${sender.username} mentioned you in a comment`,
          type: "MENTION" as const,
          link: `/movies/${movieId}?s=${slug}&t=mention`,
          senderId: sender.id,
          readAt: null,
          achievedAt: null,
          deletedAt: null,
        });

        userIds.add(user.id);

        if (user.allowEmailNotification && user.email) {
          emailNotifications.push(
            sendNotificationEmail(
              user.email,
              movieId,
              slug,
              "MENTION",
              sender.username
            )
          );
        }
      }
    }

    if (parentId) {
      const parent = await db.movieComment.findUnique({
        where: { id: parentId },
        select: {
          user: {
            select: {
              id: true,
              email: true,
              allowEmailNotification: true,
            },
          },
        },
      });

      if (
        parent &&
        parent.user.id !== sender.id &&
        !mentionedUsers.some((user) => user.id === parent.user.id)
      ) {
        notifications.push({
          userId: parent.user.id,
          message: `@${sender.username} replied to your comment`,
          type: "REPLY" as const,
          link: `/movies/${movieId}?s=${slug}&t=reply`,
          senderId: sender.id,
          readAt: null,
          achievedAt: null,
          deletedAt: null,
        });

        userIds.add(parent.user.id);

        if (parent.user.allowEmailNotification && parent.user.email) {
          emailNotifications.push(
            sendNotificationEmail(
              parent.user.email,
              movieId,
              slug,
              "REPLY",
              sender.username
            )
          );
        }
      }
    }

    if (notifications.length > 0 || emailNotifications.length > 0) {
      await db.$transaction(async (tx) => {
        if (notifications.length > 0) {
          await tx.notification.createMany({
            data: notifications,
          });
        }

        for (const userId of userIds) {
          await tx.user.update({
            where: { id: userId },
            data: {
              unreadNotificationCount: {
                increment: 1,
              },
            },
          });
        }
      });

      await Promise.all(emailNotifications);
    }
  } catch (error: unknown) {
    console.error("Failed to create notifications:", error);
    throw new Error("Failed to create notifications. Please try again.");
  }
};
