import { MAX_SEARCH_RESULTS } from "@/features/search/constants";
import { SafeSearchResult } from "@/features/search/types";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextRequest, NextResponse } from "next/server";

export async function GET(
    request: NextRequest
): Promise<NextResponse<ServerResponse>> {
    try {
        const searchParams = request.nextUrl.searchParams;
        const searchTerm = searchParams.get('q');

        if (!searchTerm || searchTerm.trim() === '') {
            return NextResponse.json(
                {
                    success: true,
                    message: "No search term provided",
                    data: [],
                },
                { status: 200 }
            );
        }

        const safeMovieTitles: SafeSearchResult[] = await db.movie.findMany({
            where: {
                title: {
                    contains: searchTerm,
                    mode: "insensitive",
                },
            },
            select: {
                movieId: true,
                title: true,
            },
            take: MAX_SEARCH_RESULTS,
        });

        return NextResponse.json(
            {
                success: true,
                message: "Search results fetched successfully",
                data: safeMovieTitles,
            },
            { status: 200 }
        );
    } catch (error: unknown) {
        console.error("Failed to fetch search results:", error);
        return NextResponse.json(
            {
                success: false,
                message: "Failed to fetch search results. Please try again.",
            },
            { status: 500 }
        );
    }
}