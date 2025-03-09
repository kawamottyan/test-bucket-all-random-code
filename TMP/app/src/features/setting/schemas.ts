import { EmailFrequency } from "@prisma/client";
import { z } from "zod";
import {
  MAX_NAME_LENGTH,
  MAX_PASSWORD_LENGTH,
  MIN_PASSWORD_LENGTH,
} from "../auth/constants";
import {
  confirmPasswordErrorMessage,
  emailErrorMessage,
  nameErrorMessages,
  passwordErrorMessages,
  passwordSchema,
} from "../auth/schemas";
import { ALLOWED_FILE_TYPES, MAX_FILE_SIZE, MAX_USERNAME_LENGTH, MIN_USERNAME_LENGTH } from "./constants";

const RESERVED_USERNAMES = ['admin', 'system', 'moderator'];

const updatePasswordErrorMessage =
  "New password must be different from current password";

const usernameErrorMessages = {
  minLength: `Password must be at least ${MAX_USERNAME_LENGTH} characters`,
  maxLength: `Password must be at most ${MAX_USERNAME_LENGTH} characters`,
};

export const AccountSchema = z.object({
  name: z
    .string()
    .min(1, { message: nameErrorMessages.minLength })
    .max(MAX_NAME_LENGTH, nameErrorMessages.maxLength),
});

export const UpdateEmailSchema = z.object({
  email: z.string().email({ message: emailErrorMessage }),
});

export const UpdatePasswordSchema = z
  .object({
    currentPassword: z
      .string()
      .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
      .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
    newPassword: passwordSchema,
    confirmNewPassword: z
      .string()
      .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
      .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
  })
  .refine((data) => data.newPassword === data.confirmNewPassword, {
    path: ["confirmPassword"],
    message: confirmPasswordErrorMessage,
  })
  .refine((data) => data.currentPassword !== data.newPassword, {
    path: ["newPassword"],
    message: updatePasswordErrorMessage,
  });

export const ProfileSchema = z.object({
  username: z
    .string()
    .min(MIN_USERNAME_LENGTH, usernameErrorMessages.minLength)
    .max(MAX_USERNAME_LENGTH, usernameErrorMessages.maxLength)
    .regex(/^[a-zA-Z0-9_-]+$/, {
      message: "Username can only contain letters, numbers, underscores, and hyphens"
    })
    .regex(/^[a-zA-Z]/, {
      message: "Username must start with a letter"
    })
    .refine(
      (username) => !RESERVED_USERNAMES.includes(username.toLowerCase()),
      {
        message: "This username is reserved and cannot be used"
      }
    )
});

export const PreferenceSchema = z.object({
  allowEmailNotification: z.boolean(),
  emailFrequency: z.nativeEnum(EmailFrequency).default(EmailFrequency.NONE),
});

export const UploadImageSchema = z.object({
  filename: z.string().min(1, "Filename is required"),
  contentType: z.string().refine(
    (value) => ALLOWED_FILE_TYPES.includes(value),
    {
      message: `Only the following file types are allowed: ${ALLOWED_FILE_TYPES.join(", ")}`,
    }
  ),
  fileSize: z.number().positive().max(
    MAX_FILE_SIZE,
    `File size must be less than ${MAX_FILE_SIZE / (1024 * 1024)}MB`
  ),
});

export const ProfileImageSchema = z.object({
  image: z.string().url("Invalid image URL format").startsWith(
    process.env.R2_PUBLIC_DOMAIN || "",
    {
      message: "Image URL must be from our storage domain",
    }
  ),
});

export type AccountFormValues = z.infer<typeof AccountSchema>;
export type UpdateEmailFormValues = z.infer<typeof UpdateEmailSchema>;
export type UpdatePasswordFormValues = z.infer<typeof UpdatePasswordSchema>;
export type ProfileFormValues = z.infer<typeof ProfileSchema>;
export type PreferenceFormValues = z.infer<typeof PreferenceSchema>;
