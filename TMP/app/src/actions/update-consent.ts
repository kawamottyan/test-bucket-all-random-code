"use server";

import { db } from "@/lib/db";
import getCurrentUser from "./get-user";

export async function updateConsent(uuid: string, trainingConsent: boolean) {
    try {
        const currentUser = await getCurrentUser();

        const result = await db.uuid.upsert({
            where: { uuid },
            update: {
                trainingConsent: trainingConsent,
                userId: currentUser?.id || undefined
            },
            create: {
                uuid,
                trainingConsent: trainingConsent,
                userId: currentUser?.id || undefined
            }
        });
        return { success: true, data: result };
    } catch (error) {
        console.error("Failed to update consent:", error);
        return { success: false, error: "Failed to update consent" };
    }
}