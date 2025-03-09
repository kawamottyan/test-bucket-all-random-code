import { EXPIRATION_TIMEOUT, MINUTES } from "@/features/auth/constants";

const expirationInMinutes = EXPIRATION_TIMEOUT / MINUTES;

export const createPasswordResetEmailTemplate = (
  email: string,
  token: string
) => ({
  subject: "Reset Your Password",
  html: `
        <p>Hi there,</p>
        <p>We received a request to reset your password. Click the link below to create a new password:</p>
        <p><a href="${process.env.NEXT_PUBLIC_APP_URL}/auth/reset-password?email=${encodeURIComponent(email)}&token=${encodeURIComponent(token)}">
            Reset Password
        </a></p>
        <p>If you didn't request this password reset, you can safely ignore this email.</p>
        <p>This link will expire in ${expirationInMinutes} minutes.</p>
        <p>Best regards,<br/>
        Masato Kawamoto from MovieAI</p>
    `,
});
