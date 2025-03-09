import getCurrentUser from "@/actions/get-user";
import { AccountSchema } from "@/features/setting/schemas";
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
          message: "Forbidden: Cannot modify other user's account",
        },
        { status: 403 }
      );
    }

    const validatedData = AccountSchema.safeParse(body);

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
      data: validatedData.data,
    });

    return NextResponse.json(
      { success: true, message: "Account updated successfully" },
      { status: 200 }
    );
  } catch (error: unknown) {
    console.error("Failed to update account:", error);
    return NextResponse.json(
      { success: false, message: "Failed to update account. Please try again" },
      { status: 500 }
    );
  }
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<RequestParams> }
): Promise<NextResponse<ServerResponse>> {
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
          message: "Forbidden: Cannot modify other user's account",
        },
        { status: 403 }
      );
    }
    await db.user.delete({
      where: {
        id: userId,
      },
    });

    return NextResponse.json(
      { success: true, message: "Account delete successfully" },
      { status: 200 }
    );
  } catch (error: unknown) {
    console.error("Failed to delete account:", error);
    return NextResponse.json(
      { success: false, message: "Failed to delete account. Please try again" },
      { status: 500 }
    );
  }
}
