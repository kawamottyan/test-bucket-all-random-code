import getCurrentUser from "@/actions/get-user";
import { SafeBookmark } from "@/features/movie-bookmark/types";
import { POSTER_BASE_URL } from "@/features/movie/constants";
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

        const bookmarks = await db.movieBookmark.findMany({
            where: {
                userId: currentUser.id,
                deletedAt: null,
            },
            select: {
                id: true,
                movieId: true,
                createdAt: true,
                deletedAt: true,
                movie: {
                    select: {
                        movieId: true,
                        title: true,
                        posterPath: true,
                        overview: true,
                    },
                },
            },
            orderBy: {
                createdAt: "desc",
            },
        });

        const safeBookmarks: SafeBookmark[] = bookmarks.map((bookmark) => ({
            ...bookmark,
            createdAt: bookmark.createdAt.toISOString(),
            deletedAt: bookmark.deletedAt?.toISOString() ?? null,
            movie: {
                ...bookmark.movie,
                posterPath: `${POSTER_BASE_URL}${bookmark.movie.posterPath}`,
            },
        }));

        return NextResponse.json(
            {
                success: true,
                message: "Bookmarks fetched successfully",
                data: safeBookmarks,
            },
            { status: 200 }
        );
    } catch (error: unknown) {
        console.error("Failed to fetch bookmarks:", error);
        return NextResponse.json(
            {
                success: false,
                message: "Failed to fetch bookmarks. Please try again.",
            },
            { status: 500 }
        );
    }
}