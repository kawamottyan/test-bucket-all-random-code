"use server";

import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const movieId = parseInt(searchParams.get('movieId') || '0', 10);
    const skip = parseInt(searchParams.get('skip') || '0', 10);
    const take = parseInt(searchParams.get('take') || '10', 10);

    const comments = await prisma.comment.findMany({
      where: {
        movieId: movieId,
        parentId: { isSet: false },
      },
      orderBy: {
        createdAt: "desc",
      },
      skip: skip,
      take: take,
      include: {
        user: {
          select: {
            username: true,
            image: true,
          },
        },
        _count: {
          select: { replies: true },
        },
      },
    });

    const safeComments = comments.map((comment) => ({
      ...comment,
      createdAt: comment.createdAt.toISOString(),
      updatedAt: comment.updatedAt.toISOString(),
    }));

    return NextResponse.json(
      { comments: safeComments },
      { status: 200 }
    );
  } catch {
    return NextResponse.json(
      { error: "An unknown error occurred. Please try again later." },
      { status: 500 }
    );
  }
}