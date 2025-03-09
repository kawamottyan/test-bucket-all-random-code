"use server";

import { db } from "@/lib/db";
import { generateUniqueUsername } from "@/lib/username";
import { ServerResponse } from "@/types";
import { LOCKOUT_DURATION, MAX_ATTEMPTS } from "../constants";
import { OTPFormValues, OTPSchema } from "../schemas";

export interface TokenWithRateLimitFields {
  attempts: number;
  lastAttempt: Date | null;
}

export async function verifyEmail(
  values: OTPFormValues
): Promise<ServerResponse> {
  try {
    const validatedFields = OTPSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid verification code format",
      };
    }

    const { email, otp } = validatedFields.data;

    const token = await db.emailVerificationToken.findFirst({
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

    const validToken = await db.emailVerificationToken.findFirst({
      where: {
        email,
        token: otp,
        expires: { gt: new Date() },
      },
    });

    if (!validToken) {
      await db.emailVerificationToken.updateMany({
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

    const username = await generateUniqueUsername();
    await db.user.update({
      where: { email },
      data: {
        emailVerified: new Date(),
        username,
      },
    });

    await db.emailVerificationToken.deleteMany({
      where: { email },
    });

    return {
      success: true,
      message: "Your email has been verified",
    };
  } catch (error: unknown) {
    console.error("Failed to verify your one-time password:", error);
    return {
      success: false,
      message: "Failed to verify your one-time password. Please try again.",
    };
  }
}

// export async function checkLockoutStatus(email: string): Promise<ServerResponse | null> {
//     const token = await db.emailVerificationToken.findFirst({
//         where: { email },
//         select: { attempts: true, lastAttempt: true }
//     });

//     if (!token) {
//         return {
//             success: false,
//             message: "Verification token not found"
//         };
//     }

//     const status = await handleRateLimiting(token);
//     if (status) return status;

//     if (token?.attempts >= MAX_ATTEMPTS) {
//         await db.emailVerificationToken.updateMany({
//             where: { email },
//             data: {
//                 attempts: 0,
//                 lastAttempt: null
//             }
//         });
//     }
//     return null;
// }

export async function handleRateLimiting(
  token: TokenWithRateLimitFields,
  email: string
): Promise<ServerResponse | null> {
  if (token.attempts >= MAX_ATTEMPTS && token.lastAttempt) {
    const timeLeft =
      LOCKOUT_DURATION - (Date.now() - token.lastAttempt.getTime());

    if (timeLeft > 0) {
      return {
        success: false,
        message: `Too many attempts. Try again in ${Math.ceil(timeLeft / 60000)} minutes`,
      };
    }

    await db.passwordResetToken.updateMany({
      where: { email },
      data: {
        attempts: 0,
        lastAttempt: null,
      },
    });
  }
  return null;
}

// export async function incrementAttempts(email: string): Promise<ServerResponse | null> {
//     await db.emailVerificationToken.updateMany({
//         where: {
//             email,
//             expires: { gt: new Date() }
//         },
//         data: {
//             attempts: { increment: 1 },
//             lastAttempt: new Date()
//         }
//     });

//     const updatedToken = await db.emailVerificationToken.findFirst({
//         where: { email }
//     });

//     if (!updatedToken) {
//         return {
//             success: false,
//             message: "Verification token not found"
//         };
//     }

//     if (updatedToken.attempts >= MAX_ATTEMPTS) {
//         const timeLeft = LOCKOUT_DURATION - (Date.now() - (updatedToken.lastAttempt?.getTime() ?? Date.now()));
//         if (timeLeft > 0) {
//             return {
//                 success: false,
//                 message: `Too many attempts. Try again in ${Math.ceil(timeLeft / 60000)} minutes`
//             };
//         }
//     }

//     return null;
// }
