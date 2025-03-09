"use server";

import getCurrentUser from "@/actions/get-user";
import { handleRateLimiting } from "@/features/auth/actions/verify";
import {
  EXPIRATION_TIMEOUT,
  OTP_TOKEN_LENGTH,
} from "@/features/auth/constants";
import { OTPFormValues, OTPSchema } from "@/features/auth/schemas";
import { db } from "@/lib/db";
import { sendVerificationEmail } from "@/lib/email";
import { generateToken } from "@/lib/tokens";
import { ServerResponse } from "@/types";
import { UpdateEmailFormValues, UpdateEmailSchema } from "../schemas";

export const initiateEmailUpdate = async ({
  email,
}: UpdateEmailFormValues): Promise<ServerResponse> => {
  try {
    const currentUser = await getCurrentUser();

    if (!currentUser || !currentUser.email || !currentUser.emailVerified) {
      throw new Error("User not authorized");
    }

    if (!currentUser.password) {
      return {
        success: false,
        message:
          "Please set up your password before changing your email address",
      };
    }

    if (currentUser.email.toLowerCase() === email.toLowerCase()) {
      return {
        success: false,
        message: "New email must be different from your current email",
      };
    }

    const emailValidation = UpdateEmailSchema.safeParse({ email });

    if (!emailValidation.success) {
      return {
        success: false,
        message: "Invalid email format",
      };
    }

    const verificationResult = await generateEmailUpdateToken(
      currentUser.id,
      emailValidation.data.email
    );

    if (!verificationResult.success) {
      return { success: false, message: "Fail to send verification code" };
    }

    return { success: true, message: "Verification email sent" };
  } catch (error: unknown) {
    console.error("Failed to send valification code:", error);
    return {
      success: false,
      message: "Failed to send valification code. Please try again.",
    };
  }
};

export async function generateEmailUpdateToken(
  userId: string,
  email: string
): Promise<ServerResponse> {
  try {
    const token = await generateToken({
      length: OTP_TOKEN_LENGTH,
      type: "numeric",
    });
    const expires = new Date(new Date().getTime() + EXPIRATION_TIMEOUT);

    await db.emailUpdateToken.deleteMany({
      where: {
        userId,
        email,
      },
    });

    await db.emailUpdateToken.create({
      data: {
        userId,
        email,
        token,
        expires,
      },
    });

    await sendVerificationEmail(email, token, "update");
    return {
      success: true,
      message: "Verification code sent to your email.",
    };
  } catch (error: unknown) {
    console.error("Failed to create verification token:", error);
    return {
      success: false,
      message: "Failed to send verification code. Please try again.",
    };
  }
}

export const verifyEmailUpdate = async (
  values: OTPFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = OTPSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid verification code format",
      };
    }

    const { email, otp } = validatedFields.data;

    const token = await db.emailUpdateToken.findFirst({
      where: {
        email,
        expires: { gt: new Date() },
      },
    });

    if (!token) {
      return { success: false, message: "No valid verification token found" };
    }

    const status = await handleRateLimiting(token, email);
    if (status) return status;

    const currentUser = await getCurrentUser();

    if (!currentUser || !currentUser.email || !currentUser.emailVerified) {
      throw new Error("User not authorized");
    }

    const validToken = await db.emailUpdateToken.findFirst({
      where: {
        email,
        token: otp,
        expires: { gt: new Date() },
      },
    });

    if (!validToken) {
      await db.emailUpdateToken.updateMany({
        where: {
          email,
          expires: { gt: new Date() },
        },
        data: {
          attempts: { increment: 1 },
          lastAttempt: new Date(),
        },
      });
      return {
        success: false,
        message: "Invalid or expired verification code",
      };
    }

    await db.user.update({
      where: { id: currentUser.id },
      data: {
        email,
        emailVerified: new Date(),
      },
    });

    await db.emailUpdateToken.deleteMany({
      where: { id: currentUser.id },
    });

    return { success: true, message: "Your new email has been verified" };
  } catch (error: unknown) {
    console.error("Failed to verify your one-time password:", error);
    return {
      success: false,
      message: "Failed to verify your one-time password. Please try again.",
    };
  }
};
