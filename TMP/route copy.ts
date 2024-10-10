"use server";

import getCurrentUser from "@/action/getCurrentUser";
import prisma from "@/lib/db";
import { NextResponse } from "next/server";

interface IParams {
  movieId: string;
}

export async function POST(request: Request, { params }: { params: IParams }) {
  try {
    const currentUser = await getCurrentUser();

    if (!currentUser) {
      return NextResponse.error();
    }

    const { movieId } = params;
    const { content } = await request.json();

    const comment = await prisma.comment.create({
      data: {
        userId: currentUser.id,
        movieId: parseInt(movieId, 10),
        content: content,
      },
      include: {
        user: {
          select: {
            username: true,
            image: true,
          },
        },
      },
    });

    const safeComment = {
      ...comment,
      createdAt: comment.createdAt.toISOString(),
      updatedAt: comment.updatedAt.toISOString(),
    };

    return NextResponse.json(
      { comment: safeComment },
      { status: 201 }
    );
  } catch {
    return NextResponse.json(
      { error: "An unknown error occurred. Please try again later." },
      { status: 500 }
    );
  }
}