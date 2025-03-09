import getCurrentUser from "@/actions/get-user";
import { db } from "@/lib/db";
import { NextResponse } from "next/server";

export async function PATCH() {
  try {
    const currentUser = await getCurrentUser();

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    const now = new Date();

    await db.$transaction([
      db.notification.updateMany({
        where: {
          userId: currentUser.id,
          achievedAt: null,
        },
        data: {
          readAt: now,
          achievedAt: now,
        },
      }),
      db.user.update({
        where: { id: currentUser.id },
        data: { unreadNotificationCount: 0 },
      }),
    ]);

    return NextResponse.json(
      {
        success: true,
        message: "Notification updated successfully",
      },
      { status: 200 }
    );
  } catch (error: unknown) {
    console.error("Failed to update notifications:", error);
    return NextResponse.json(
      {
        success: false,
        message: "Failed to update notifications. Please try again.",
      },
      { status: 500 }
    );
  }
}
