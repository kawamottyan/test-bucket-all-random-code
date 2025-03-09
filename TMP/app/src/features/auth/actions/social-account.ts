"use server";

import getCurrentUser from "@/actions/get-user";
import { signIn } from "@/lib/auth";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types/index";

export const linkSocialAccount = async (
  provider: string
): Promise<ServerResponse> => {
  await signIn(provider);
  return {
    success: true,
    message: `Successfully link ${provider} account`,
  };
};

export const unlinkSocialAccount = async (
  provider: string
): Promise<ServerResponse> => {
  try {
    const user = await getCurrentUser();

    if (!user) {
      return {
        success: false,
        message: "Unauthorized: Please log in",
      };
    }

    const userHasPassword = await db.user.findUnique({
      where: { id: user.id },
      select: { password: true },
    });

    if (!userHasPassword?.password) {
      return {
        success: false,
        message:
          "Please set up a password first before unlinking social accounts.",
      };
    }

    const accounts = await db.account.findMany({
      where: {
        userId: user.id,
        provider: provider,
      },
    });

    if (accounts.length === 0) {
      return {
        success: false,
        message: "Account not found",
      };
    }

    await db.account.deleteMany({
      where: {
        userId: user.id,
        provider: provider,
      },
    });

    return {
      success: true,
      message: `Successfully unlinked ${provider} account`,
    };
  } catch (error) {
    console.error("Failed to unlink social account:", error);
    return {
      success: false,
      message: "Failed to unlink social account",
    };
  }
};
