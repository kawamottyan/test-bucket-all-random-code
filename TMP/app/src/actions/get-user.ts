"use server";

import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { User } from "@prisma/client";

export const getUserByEmail = async (email: string): Promise<User | null> => {
  try {
    const user = await db.user.findUnique({ where: { email } });

    return user;
  } catch {
    return null;
  }
};

export const getUserById = async (id: string): Promise<User | null> => {
  try {
    const user = await db.user.findUnique({ where: { id } });

    return user;
  } catch {
    return null;
  }
};

export default async function getCurrentUser(): Promise<User | null> {
  try {
    const session = await auth();

    if (!session?.user?.email) {
      return null;
    }

    const currentUser = await db.user.findUnique({
      where: {
        email: session.user.email as string,
      },
    });

    if (!currentUser) {
      throw new Error("User not authorized");
    }

    if (!currentUser.emailVerified) {
      throw new Error("User not verified");
    }

    return currentUser;
  } catch (error: unknown) {
    console.error("Failed to fetch current user:", error);
    throw new Error("Failed to fetch current user. Please try again.");
  }
}
