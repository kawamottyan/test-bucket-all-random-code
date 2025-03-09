
import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { NextResponse } from "next/server";

export async function GET() {
    try {
        const session = await auth();
        if (!session?.user?.email) {
            return NextResponse.json(null);
        }

        const user = await db.user.findUnique({
            where: {
                email: session.user.email
            },
            select: {
                id: true,
                name: true,
                password: true,
                email: true,
                emailVerified: true,
                image: true,
                role: true,
                isPrivate: true,
                username: true,
                bio: true,
                allowAdultMovie: true,
                allowEmailNotification: true,
                emailFrequency: true,
                unreadNotificationCount: true,
                bookmarkCount: true,
                reviewCount: true,
                createdAt: true,
                updatedAt: true,
                accounts: {
                    select: {
                        provider: true,
                    },
                },
            }
        });

        if (!user) {
            return NextResponse.json(null);
        }

        const { password, ...userWithoutPassword } = user;

        return NextResponse.json({
            ...userWithoutPassword,
            hasPassword: password !== null,
            emailVerified: user.emailVerified?.toISOString() || null,
            createdAt: user.createdAt.toISOString(),
            updatedAt: user.updatedAt.toISOString(),
            accounts: user.accounts,
        });
    } catch (error) {
        console.error("Failed to fetch user:", error);
        return NextResponse.json(
            { error: "Failed to fetch user" },
            { status: 500 }
        );
    }
}