import getCurrentUser from "@/actions/get-user";
import redis from "@/db/redis";
import { cookies, headers } from "next/headers";
import { UAParser } from "ua-parser-js";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
    const [currentUser, clientInfoResult, cookieStore, headersList] = await Promise.all([
        getCurrentUser(),
        request.json().catch(() => ({})),
        cookies(),
        headers()
    ]);

    const clientInfo = clientInfoResult;
    const uuid = cookieStore.get('uuid')?.value;
    const sessionId = cookieStore.get('session_id')?.value;

    if (!sessionId) {
        return NextResponse.json({
            success: false,
            message: 'No session ID'
        }, { status: 400 });
    }

    const existing = await redis.get(`session:${sessionId}`);
    if (existing) {
        return NextResponse.json({
            success: true,
            message: 'Session already exists'
        });
    }

    const userAgent = headersList.get('user-agent') || '';
    const parser = new UAParser(userAgent);
    const ip = headersList.get("x-forwarded-for") || "127.0.0.1";

    const sessionInfo = {
        uuid: uuid || undefined,
        user_id: currentUser?.id || undefined,

        device: parser.getDevice().type || 'desktop',
        ip,
        browser: `${parser.getBrowser().name || 'unknown'} ${parser.getBrowser().version || ''}`,
        os: `${parser.getOS().name || 'unknown'} ${parser.getOS().version || ''}`,
        created_at: new Date().toISOString(),

        timezone: clientInfo.timezone || 'Unknown',
        language: clientInfo.language,
        screen_resolution: clientInfo.screenWidth && clientInfo.screenHeight ?
            `${clientInfo.screenWidth}x${clientInfo.screenHeight}` : undefined,
        color_depth: clientInfo.colorDepth,
        window_size: clientInfo.windowWidth && clientInfo.windowHeight ?
            `${clientInfo.windowWidth}x${clientInfo.windowHeight}` : undefined,
        device_pixel_ratio: clientInfo.devicePixelRatio,
        cookies_enabled: clientInfo.cookiesEnabled,
        do_not_track: clientInfo.doNotTrack === "1" || clientInfo.doNotTrack === "yes",
        platform: clientInfo.platform,
        vendor: clientInfo.vendor,
        referrer: clientInfo.referrer,
        connection_type: clientInfo.connectionType,
    };

    await redis.set(`session:${sessionId}`, JSON.stringify(sessionInfo), {
        ex: 60 * 60 * 24 * 7
    });

    return NextResponse.json({
        success: true,
        message: 'Session created successfully'
    });
}