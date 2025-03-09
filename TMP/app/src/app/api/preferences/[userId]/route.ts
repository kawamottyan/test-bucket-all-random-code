import getCurrentUser from "@/actions/get-user";
import { PreferenceSchema } from "@/features/setting/schemas";
import { db } from "@/lib/db";
import { ServerResponse } from "@/types";
import { NextResponse } from "next/server";

interface RequestParams {
  userId: string;
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
  try {
    const [body, { userId }, currentUser] = await Promise.all([
      request.json(),
      Promise.resolve(params),
      getCurrentUser(),
    ]);

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    if (currentUser.id !== userId) {
      return NextResponse.json(
        {
          success: false,
          message: "Forbidden: Cannot modify other user's preferences",
        },
        { status: 403 }
      );
    }

    const validatedData = PreferenceSchema.safeParse(body);

    if (!validatedData.success) {
      return NextResponse.json(
        { success: false, message: "Invalid input data" },
        { status: 400 }
      );
    }

    await db.user.update({
      where: {
        id: userId,
      },
      data: {
        ...validatedData.data,
      },
    });

    return NextResponse.json(
      { success: true, message: "Preferences updated successfully" },
      { status: 200 }
    );
  } catch (error: unknown) {
    console.error("Failed to update preferences:", error);
    return NextResponse.json(
      { success: false, message: "Failed to update preferences" },
      { status: 500 }
    );
  }
}
