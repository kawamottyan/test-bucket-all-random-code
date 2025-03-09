import { createNotificationEmailTemplate } from "@/templates/notification-email";
import { createPasswordResetEmailTemplate } from "@/templates/password-reset-email";
import {
  createEmailUpdateTemplate,
  createEmailVerifyTemplate,
} from "@/templates/verification-email";
import { NotificationType } from "@prisma/client";
import { Resend } from "resend";

const resend = new Resend(process.env.RESEND_API_KEY);
const fromEmail = process.env.FROM_EMAIL as string;

export const sendVerificationEmail = async (
  email: string,
  token: string,
  type: "verification" | "update"
) => {
  const template =
    type === "verification"
      ? createEmailVerifyTemplate(email, token)
      : createEmailUpdateTemplate(email, token);

  await resend.emails.send({
    from: fromEmail,
    to: email,
    subject: template.subject,
    html: template.html,
  });
};

export const sendPasswordResetEmail = async (email: string, token: string) => {
  const { subject, html } = createPasswordResetEmailTemplate(email, token);

  await resend.emails.send({
    from: fromEmail,
    to: email,
    subject,
    html,
  });
};

export const sendNotificationEmail = async (
  email: string,
  movieId: number,
  slug: string,
  notificationType: NotificationType,
  senderUsername: string | null
) => {
  const { subject, html } = createNotificationEmailTemplate({
    movieId,
    slug,
    notificationType,
    senderUsername,
  });

  await resend.emails.send({
    from: fromEmail,
    to: email,
    subject,
    html,
  });
};
