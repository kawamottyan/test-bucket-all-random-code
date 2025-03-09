import getCurrentUser from "@/actions/get-user";
import { SafeReview } from "@/features/movie-review/types";
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

        const reviews = await db.movieReview.findMany({
            where: {
                userId: currentUser.id,
                deletedAt: null,
            },
            select: {
                id: true,
                movieId: true,
                rating: true,
                note: true,
                watchedAt: true,
                deletedAt: true,
                movie: {
                    select: {
                        movieId: true,
                        title: true,
                        releaseDate: true,
                    },
                },
            },
            orderBy: {
                createdAt: "desc",
            },
        });

        const safeReviews: SafeReview[] = reviews.map((review) => ({
            ...review,
            watchedAt: review.watchedAt.toISOString(),
            deletedAt: review.deletedAt?.toISOString() ?? null,
            movie: {
                ...review.movie,
                releaseDate: review.movie.releaseDate?.toISOString() || null,
            },
        }));

        return NextResponse.json(
            {
                success: true,
                message: "Reviews fetched successfully",
                data: safeReviews,
            },
            { status: 200 }
        );
    } catch (error: unknown) {
        console.error("Failed to fetch reviews:", error);
        return NextResponse.json(
            {
                success: false,
                message: "Failed to fetch reviews. Please try again.",
            },
            { status: 500 }
        );
    }
}