import { EXPIRATION_TIMEOUT, MINUTES } from "@/features/auth/constants";

const expirationInMinutes = EXPIRATION_TIMEOUT / MINUTES;

export const createEmailVerifyTemplate = (email: string, token: string) => ({
  subject: "Verify Your Email Address",
  html: `
    <p>Hi there,</p>
    <p>Thank you for signing up! Please use the following code to verify your email address:</p>
    <p><strong>${token}</strong></p>
    <p>Or click the link below to verify your email:</p>
    <p><a href="${process.env.NEXT_PUBLIC_APP_URL}/auth/verify-email?email=${encodeURIComponent(email)}">Verify Email</a></p>
    <p>If you didn't request this code, you can safely ignore this email.</p>
    <p>This link will expire in ${expirationInMinutes} minutes.</p>
    <p>Best regards,<br/>
    Masato Kawamoto from MovieAI</p>
  `,
});

export const createEmailUpdateTemplate = (email: string, token: string) => ({
  subject: "Verify Your New Email Address",
  html: `
    <p>Hi there,</p>
    <p>Please use the following code to verify your new email address:</p>
    <p><strong>${token}</strong></p>
    <p>Or click the link below to verify your email change:</p>
    <p><a href="${process.env.NEXT_PUBLIC_APP_URL}/setting/verify-email-update?email=${encodeURIComponent(email)}">Verify Email Change</a></p>
    <p>If you didn't request to change your email, please secure your account immediately.</p>
    <p>This link will expire in ${expirationInMinutes} minutes.</p>
    <p>Best regards,<br/>
    Masato Kawamoto from MovieAI</p>
  `,
});
