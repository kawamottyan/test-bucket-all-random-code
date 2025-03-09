'use server'

import redis from '@/db/redis';
import { InteractionType, SafeInteractionLog, ServerResponse } from '@/types';
import { cookies } from 'next/headers';

interface LogInteractionParams {
    interactionType: InteractionType;
    itemId?: string;
    watchTime?: string;
    query?: string;
    uuid?: string;
    timestamp: string
    index?: string
}

export async function logInteraction({
    interactionType,
    itemId,
    watchTime,
    query,
    uuid,
    timestamp,
    index
}: LogInteractionParams): Promise<ServerResponse> {
    const cookieStore = await cookies();
    const session_id = cookieStore.get('session_id')?.value;

    if (!session_id) {
        throw new Error("Missing session id in cookies");
    }
    try {
        const interactionKey = `interaction:${session_id}:${Date.now()}`;

        const interactionData: SafeInteractionLog = {
            uuid,
            interaction_type: interactionType,
            item_id: itemId,
            watch_time: watchTime,
            query: query,
            created_at: String(Date.now()),
            index: index,
            local_timestamp: timestamp
        };

        const cleanedData = Object.fromEntries(
            Object.entries(interactionData).filter(([, value]) => value !== undefined)
        );

        await redis.hset(interactionKey, cleanedData);
        await redis.expire(interactionKey, 60 * 60 * 24 * 7);

        return {
            success: true,
            message: "Interaction logged successfully"
        };
    } catch (error) {
        console.error("Fail to save interaction:", error);
        return {
            success: false,
            message: "Fail to save interaction"
        };
    }
}