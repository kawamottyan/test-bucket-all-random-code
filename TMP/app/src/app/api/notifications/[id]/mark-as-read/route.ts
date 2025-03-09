import getCurrentUser from "@/actions/get-user";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  id: string;
}

export async function PATCH(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [{ id }, currentUser] = await Promise.all([
      Promise.resolve(params),
      getCurrentUser(),
    ]);

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    const existingNotification = await db.notification.findUnique({
      where: { id },
    });

    if (!existingNotification || existingNotification.deletedAt) {
      return NextResponse.json(
        { success: false, message: "Notification not found" },
        { status: 404 }
      );
    }

    await db.$transaction([
      db.notification.update({
        where: { id: existingNotification.id },
        data: { readAt: new Date() },
      }),
      db.user.update({
        where: { id: currentUser.id },
        data: { unreadNotificationCount: { decrement: 1 } },
      }),
    ]);

    return NextResponse.json({
      success: true,
      message: "Notification mark as read successfully",
    });
  } catch (error) {
    console.error("Failed to mark notification as read:", error);
    return NextResponse.json(
      { success: false, message: "Failed to mark notification as read" },
      { status: 500 }
    );
  }
}
