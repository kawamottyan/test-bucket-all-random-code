import { z } from "zod";
import {
  MAX_NAME_LENGTH,
  MAX_PASSWORD_LENGTH,
  MIN_PASSWORD_LENGTH,
  OTP_TOKEN_LENGTH,
} from "./constants";

export const nameErrorMessages = {
  minLength: "Please enter your name",
  maxLength: `Name must be less than ${MAX_NAME_LENGTH} characters`,
};

export const passwordErrorMessages = {
  minLength: `Password must be at least ${MIN_PASSWORD_LENGTH} characters`,
  maxLength: `Password must be at most ${MAX_PASSWORD_LENGTH} characters`,
  uppercase: "Password must contain at least one uppercase letter",
  lowercase: "Password must contain at least one lowercase letter",
  number: "Password must contain at least one number",
  specialChar:
    "Password must contain at least one special character (!@#$%^&*-_+=?/\\|~`<>{}[]())",
};

export const emailErrorMessage = "Please enter a valid email";
export const confirmPasswordErrorMessage = "Passwords do not match";
const otpErrorMessage = "Please enter a valid code";

export const passwordSchema = z
  .string()
  .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
  .max(MAX_PASSWORD_LENGTH, { message: passwordErrorMessages.maxLength })
  .refine((password) => /[A-Z]/.test(password), {
    message: passwordErrorMessages.uppercase,
  })
  .refine((password) => /[a-z]/.test(password), {
    message: passwordErrorMessages.lowercase,
  })
  .refine((password) => /[0-9]/.test(password), {
    message: passwordErrorMessages.number,
  })
  .refine((password) => /[!@#$%^&*\-_+=?/\\|~`<>{}[\]()]/.test(password), {
    message: passwordErrorMessages.specialChar,
  });

export const RegisterSchema = z
  .object({
    name: z
      .string()
      .min(1, { message: nameErrorMessages.minLength })
      .max(MAX_NAME_LENGTH, nameErrorMessages.maxLength),
    email: z.string().email({ message: emailErrorMessage }),
    password: passwordSchema,
    confirmPassword: z
      .string()
      .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
      .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
  })
  .refine((data) => data.password === data.confirmPassword, {
    path: ["confirmPassword"],
    message: confirmPasswordErrorMessage,
  });

export const OTPSchema = z.object({
  email: z.string().email({ message: emailErrorMessage }),
  otp: z.string().length(OTP_TOKEN_LENGTH, otpErrorMessage),
});

export const ResendOTPSchema = z.object({
  email: z.string().email({ message: emailErrorMessage }),
});

export const InitiatePasswordResetSchema = z.object({
  email: z.string().email({ message: emailErrorMessage }),
});

export const SetPasswordSchema = z
  .object({
    password: passwordSchema,
    confirmPassword: z
      .string()
      .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
      .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
  })
  .refine((data) => data.password === data.confirmPassword, {
    path: ["confirmPassword"],
    message: confirmPasswordErrorMessage,
  });

export const ResetPasswordSchema = z
  .object({
    email: z.string().email(),
    token: z.string(),
    password: passwordSchema,
    confirmPassword: z
      .string()
      .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
      .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
  })
  .refine((data) => data.password === data.confirmPassword, {
    path: ["confirmPassword"],
    message: confirmPasswordErrorMessage,
  });

export const LoginSchema = z.object({
  email: z.string().email({ message: emailErrorMessage }),
  password: z
    .string()
    .min(MIN_PASSWORD_LENGTH, { message: passwordErrorMessages.minLength })
    .max(MAX_PASSWORD_LENGTH, passwordErrorMessages.maxLength),
});

export type RegisterFormValues = z.infer<typeof RegisterSchema>;
export type OTPFormValues = z.infer<typeof OTPSchema>;
export type ResendOTPFormValues = z.infer<typeof ResendOTPSchema>;
export type InitiatePasswordResetFormValues = z.infer<
  typeof InitiatePasswordResetSchema
>;
export type SetPasswordFormValues = z.infer<typeof SetPasswordSchema>;
export type ResetPasswordFormValues = z.infer<typeof ResetPasswordSchema>;
export type LoginFormValues = z.infer<typeof LoginSchema>;
