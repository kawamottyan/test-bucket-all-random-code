import getCurrentUser from "@/actions/get-user";
import { ProfileImageSchema } from "@/features/setting/schemas";
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
          message: "Forbidden: Cannot modify other user's profile",
        },
        { status: 403 }
      );
    }

    const result = ProfileImageSchema.safeParse(body);
    if (!result.success) {
      const errorMessage = result.error.errors.map(err => err.message).join(", ");
      return NextResponse.json(
        {
          success: false,
          message: errorMessage,
        },
        { status: 400 }
      );
    }

    const { image } = result.data;

    await db.user.update({
      where: { id: currentUser.id },
      data: { image },
    });

    return NextResponse.json(
      { success: true, message: "Profile image updated successfully" },
      { status: 200 }
    );
  } catch (error: unknown) {
    console.error("Failed to update profile image:", error);
    return NextResponse.json(
      { success: false, message: "Failed to update profile image" },
      { status: 500 }
    );
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
) {
  try {
    const [{ userId }, currentUser] = await Promise.all([
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
          message: "Forbidden: Cannot modify other user's profile",
        },
        { status: 403 }
      );
    }

    await db.user.update({
      where: { id: userId },
      data: { image: null },
    });

    return NextResponse.json({
      success: true,
      message: "Profile image removed successfully",
    });
  } catch (error) {
    console.error("Failed to remove profile image:", error);
    return NextResponse.json(
      { success: false, message: "Failed to remove profile image" },
      { status: 500 }
    );
  }
}
