import { NotificationType } from "@prisma/client";

interface CreateNotificationEmailProps {
  movieId: number;
  slug: string;
  notificationType: NotificationType;
  senderUsername: string | null;
}

export const createNotificationEmailTemplate = ({
  movieId,
  slug,
  notificationType,
  senderUsername,
}: CreateNotificationEmailProps) => {
  const typeSpecificContent = {
    MENTION: {
      subject: `${senderUsername ? `@${senderUsername}` : "Someone"} mentioned you in a comment on MovieAI`,
      message: `mentioned you in a comment`,
    },
    REPLY: {
      subject: `${senderUsername ? `@${senderUsername}` : "Someone"} replied to your comment on MovieAI`,
      message: `replied to your comment`,
    },
  };

  const content = typeSpecificContent[notificationType];
  const link = `${process.env.NEXT_PUBLIC_APP_URL}/movies/${movieId}?s=${slug}&t=${notificationType.toLowerCase()}`;

  return {
    subject: content.subject,
    html: `
        <p>Hi there,</p>
        <p>@${senderUsername} ${content.message}. Click the link below to view:</p>
        <p><a href="${link}">
            View Comment
        </a></p>
        <p>If you didn't expect this notification, you can manage your notification settings in your account preferences.</p>
        <p>Best regards,<br/>
        Masato Kawamoto from MovieAI</p>
    `,
  };
};
