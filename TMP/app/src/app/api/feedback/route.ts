import getCurrentUser from "@/actions/get-user";
import { FeedbackFormSchema } from "@/features/support/schemas";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

export async function POST(
  request: Request
): Promise<NextResponse<ServerResponse>> {
  try {
    const currentUser = await getCurrentUser();

    const body = await request.json();
    const validatedData = FeedbackFormSchema.safeParse(body);

    if (!validatedData.success) {
      return NextResponse.json(
        { success: false, message: "Invalid input data" },
        { status: 400 }
      );
    }

    const { type, email, title, description } = validatedData.data;

    if (type === "SUPPORT" && !currentUser && !email) {
      return NextResponse.json(
        { success: false, message: "Email is required for support requests" },
        { status: 400 }
      );
    }

    await db.feedback.create({
      data: {
        type,
        title,
        description,
        userId: currentUser?.id,
        email: currentUser ? currentUser.email : email,
      },
    });

    return NextResponse.json(
      { success: true, message: "Feedback created successfully" },
      { status: 200 }
    );
  } catch (error) {
    console.error("Failed to create feedback:", error);
    return NextResponse.json(
      { success: false, message: "Failed to create feedback" },
      { status: 500 }
    );
  }
}
