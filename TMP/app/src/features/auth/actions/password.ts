"use server";

import {
  UpdatePasswordFormValues,
  UpdatePasswordSchema,
} from "@/features/setting/schemas";
import { db } from "@/lib/db";
import { sendPasswordResetEmail } from "@/lib/email";
import { generateToken } from "@/lib/tokens";
import { ServerResponse } from "@/types/index";
import bcrypt from "bcryptjs";
import getCurrentUser, { getUserByEmail } from "../../../actions/get-user";
import { EXPIRATION_TIMEOUT, PASSWORD_RESET_TOKEN_LENGTH } from "../constants";
import {
  InitiatePasswordResetFormValues,
  InitiatePasswordResetSchema,
  ResetPasswordFormValues,
  ResetPasswordSchema,
  SetPasswordFormValues,
  SetPasswordSchema,
} from "../schemas";
import { handleRateLimiting } from "./verify";

export const initiatePasswordReset = async (
  values: InitiatePasswordResetFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = InitiatePasswordResetSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid request",
      };
    }

    const { email } = validatedFields.data;
    const existingUser = await getUserByEmail(email);

    if (!existingUser) {
      return {
        success: false,
        message: "Email not found",
      };
    }

    if (!existingUser.emailVerified) {
      return {
        success: false,
        message: "Email not verified",
      };
    }

    if (!existingUser.password) {
      return {
        success: false,
        message: "Password not set. Please login with social account.",
      };
    }

    const passwordResetToken = await generatePasswordResetToken(email);

    if (!passwordResetToken.success) {
      return passwordResetToken;
    }

    return {
      success: true,
      message: "Email sent. Please check your email for a password reset code.",
    };
  } catch (error: unknown) {
    console.error("Failed to send reset email:", error);
    return {
      success: false,
      message: "Failed to send reset email. Please try again.",
    };
  }
};

export async function generatePasswordResetToken(
  email: string
): Promise<ServerResponse> {
  try {
    const token = await generateToken({
      length: PASSWORD_RESET_TOKEN_LENGTH,
      type: "urlsafe",
    });
    const expires = new Date(new Date().getTime() + EXPIRATION_TIMEOUT);

    await db.passwordResetToken.deleteMany({
      where: { email },
    });

    await db.passwordResetToken.create({
      data: {
        email,
        token,
        expires,
      },
    });

    await sendPasswordResetEmail(email, token);
    return {
      success: true,
      message: "Password reset code sent to your email.",
    };
  } catch (error) {
    console.error("Failed to create password reset token:", error);
    return {
      success: false,
      message: "Failed to send password reset code. Please try again.",
    };
  }
}

export const resetPassword = async (
  values: ResetPasswordFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = ResetPasswordSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid request",
      };
    }

    const { email, token, password } = validatedFields.data;

    const resetToken = await db.passwordResetToken.findFirst({
      where: {
        email,
        expires: {
          gt: new Date(),
        },
      },
    });

    if (!resetToken) {
      return { success: false, message: "No valid reset token found" };
    }

    const status = await handleRateLimiting(resetToken, email);
    if (status) return status;

    const validToken = await db.passwordResetToken.findFirst({
      where: {
        email,
        token,
        expires: {
          gt: new Date(),
        },
      },
    });

    if (!validToken) {
      await db.passwordResetToken.updateMany({
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
        message: "Invalid or expired reset token",
      };
    }

    const existingUser = await getUserByEmail(email);

    if (!existingUser) {
      return { success: false, message: "User not exist" };
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    await db.user.update({
      where: { email },
      data: {
        password: hashedPassword,
      },
    });

    await db.passwordResetToken.deleteMany({
      where: { email },
    });

    return { success: true, message: "Password reset successfully" };
  } catch (error: unknown) {
    console.error("Failed to reset password:", error);
    return {
      success: false,
      message: "Failed to reset password. Please try again.",
    };
  }
};

// export async function validatePasswordResetToken(
//   email: string,
//   token: string
// ): Promise<boolean> {
//   try {
//     const resetToken = await db.passwordResetToken.findFirst({
//       where: {
//         email,
//         token: token,
//         expires: {
//           gt: new Date(),
//         },
//       },
//     });

//     return !!resetToken;
//   } catch (error) {
//     console.error("Failed to validate reset token:", error);
//     return false;
//   }
// }

// export async function setPassword(
//   email: string,
//   password: string
// ): Promise<boolean> {
//   try {
//     const hashedPassword = await bcrypt.hash(password, 10);
//     await db.user.update({
//       where: { email },
//       data: {
//         password: hashedPassword,
//       },
//     });
//     return true;
//   } catch (error) {
//     console.error("Failed to set password:", error);
//     return false;
//   }
// }

// export async function deletePasswordResetToken(email: string): Promise<void> {
//   try {
//     await db.passwordResetToken.deleteMany({
//       where: { email },
//     });
//   } catch (error) {
//     console.error("Failed to invalidate reset token:", error);
//   }
// }

export const setInitialPassword = async (
  values: SetPasswordFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = SetPasswordSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid password format",
      };
    }

    const { password } = validatedFields.data;

    const currentUser = await getCurrentUser();

    if (!currentUser) {
      throw new Error("User not authorized");
    }

    if (!currentUser.email) {
      throw new Error("Email not found");
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    await db.user.update({
      where: { id: currentUser.id },
      data: {
        password: hashedPassword,
      },
    });

    return { success: true, message: "Password set successfully" };
  } catch (error: unknown) {
    console.error("Failed to set password:", error);
    return {
      success: false,
      message: "Failed to update password. Please try again.",
    };
  }
};

export const updatePassword = async (
  values: UpdatePasswordFormValues
): Promise<ServerResponse> => {
  try {
    const validatedFields = UpdatePasswordSchema.safeParse(values);

    if (!validatedFields.success) {
      return {
        success: false,
        message: "Invalid password format",
      };
    }

    const { currentPassword, newPassword } = validatedFields.data;

    const currentUser = await getCurrentUser();

    if (!currentUser || !currentUser.password) {
      throw new Error("User not authorized");
    }

    const isCurrentPasswordValid = await bcrypt.compare(
      currentPassword,
      currentUser.password
    );

    if (!isCurrentPasswordValid) {
      return {
        success: false,
        message: "Current password is incorrect",
      };
    }

    const hashedPassword = await bcrypt.hash(newPassword, 10);

    await db.user.update({
      where: {
        id: currentUser.id,
      },
      data: {
        password: hashedPassword,
      },
    });

    return { success: true, message: "Password update successfully" };
  } catch (error: unknown) {
    console.error("Failed to update password:", error);
    return {
      success: false,
      message: "Failed to update password. Please try again.",
    };
  }
};
