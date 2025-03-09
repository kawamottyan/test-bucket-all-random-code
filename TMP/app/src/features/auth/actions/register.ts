"use server";

import { db } from "@/lib/db";
import { sendVerificationEmail } from "@/lib/email";
import { generateToken } from "@/lib/tokens";
import { ServerResponse } from "@/types";
import bcrypt from "bcryptjs";
import { getUserByEmail } from "../../../actions/get-user";
import { EXPIRATION_TIMEOUT, OTP_TOKEN_LENGTH } from "../constants";
import {
  RegisterFormValues,
  RegisterSchema,
  ResendOTPFormValues,
  ResendOTPSchema,
} from "../schemas";
import { handleRateLimiting } from "./verify";

export const register = async (
  values: RegisterFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = RegisterSchema.safeParse(values);

    if (!validatedFields.success) {
      return { success: false, message: "Invalid request" };
    }

    const { email, password, name } = validatedFields.data;
    const existingUser = await getUserByEmail(email);

    if (existingUser) {
      return { success: false, message: "Email is already in use" };
    }

    const verificationResult = await generateEmailVerificationToken(email);

    if (!verificationResult.success) {
      return { success: false, message: "Fail to send verification code" };
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    await db.user.create({
      data: {
        name,
        email,
        password: hashedPassword,
      },
    });

    return {
      success: true,
      message:
        "Account created. Please check your email for a verification code.",
    };
  } catch (error: unknown) {
    console.error("Failed to create your account:", error);
    return {
      success: false,
      message: "Failed to create your account. Please try again.",
    };
  }
};

export async function generateEmailVerificationToken(
  email: string
): Promise<ServerResponse> {
  try {
    const token = await generateToken({
      length: OTP_TOKEN_LENGTH,
      type: "numeric",
    });
    const expires = new Date(new Date().getTime() + EXPIRATION_TIMEOUT);

    await db.emailVerificationToken.deleteMany({
      where: { email },
    });

    await db.emailVerificationToken.create({
      data: {
        email,
        token,
        expires,
      },
    });

    await sendVerificationEmail(email, token, "verification");
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

export const resendVerificationToken = async (
  values: ResendOTPFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = ResendOTPSchema.safeParse(values);

    if (!validatedFields.success) {
      return { success: false, message: "Invalid request" };
    }

    const { email } = validatedFields.data;

    const token = await db.emailVerificationToken.findFirst({
      where: {
        email,
        expires: {
          gt: new Date(),
        },
      },
    });

    if (token) {
      const status = await handleRateLimiting(token, email);
      if (status) return status;
    }

    const user = await getUserByEmail(email);

    if (!user) {
      return {
        success: false,
        message: "User not registered",
      };
    }

    if (user.emailVerified) {
      return {
        success: false,
        message: "Email is already verified",
      };
    }

    const verificationResult = await generateEmailVerificationToken(email);

    if (!verificationResult.success) {
      return { success: false, message: "Failed to resend verification token" };
    }

    return {
      success: true,
      message: "Resend verification code to your email.",
    };
  } catch (error: unknown) {
    console.error("Failed to resend verification token:", error);
    return {
      success: false,
      message: "Failed to resend verification code. Please try again.",
    };
  }
};
