import getCurrentUser from "@/actions/get-user";
import { SafeNotification } from "@/features/notification/types";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

export async function GET(): Promise<NextResponse<ServerResponse>> {
    try {
        const currentUser = await getCurrentUser();

        if (!currentUser) {
            return NextResponse.json(
                {
                    success: false,
                    message: "Unauthorized: Please log in",
                },
                { status: 401 }
            );
        }

        const notifications = await db.notification.findMany({
            where: {
                userId: currentUser.id,
                achievedAt: null,
                deletedAt: null,
            },
            orderBy: {
                createdAt: "desc",
            },
            select: {
                id: true,
                message: true,
                type: true,
                link: true,
                readAt: true,
                achievedAt: true,
                deletedAt: true,
                createdAt: true,
            },
        });

        if (!notifications) {
            return NextResponse.json(
                {
                    success: true,
                    message: "No notifications found",
                    data: [],
                },
                { status: 200 }
            );
        }

        const safeNotifications: SafeNotification[] = notifications.map(
            (notification) => ({
                ...notification,
                createdAt: notification.createdAt.toISOString(),
                readAt: notification.readAt?.toISOString() ?? null,
                achievedAt: notification.achievedAt?.toISOString() ?? null,
                deletedAt: notification.deletedAt?.toISOString() ?? null,
            })
        );

        return NextResponse.json(
            {
                success: true,
                message: "Notifications fetched successfully",
                data: safeNotifications,
            },
            { status: 200 }
        );
    } catch (error: unknown) {
        console.error("Failed to fetch notifications:", error);

        return NextResponse.json(
            {
                success: false,
                message: "Failed to fetch notifications. Please try again.",
            },
            { status: 500 }
        );
    }
}