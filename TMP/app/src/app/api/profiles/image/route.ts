import getCurrentUser from "@/actions/get-user";
import { UploadImageSchema } from "@/features/setting/schemas";
import { ServerResponse } from "@/types";
import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { NextResponse } from "next/server";

const BUCKET_NAME = process.env.R2_BUCKET_NAME!;
const R2 = new S3Client({
  region: "auto",
  endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID!,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
  },
});

export async function POST(
  req: Request
): Promise<NextResponse<ServerResponse>> {
  try {
    const currentUser = await getCurrentUser();

    if (!currentUser) {
      return NextResponse.json(
        { success: false, message: "Unauthorized: Please log in" },
        { status: 401 }
      );
    }

    const body = await req.json();

    const result = UploadImageSchema.safeParse(body);
    if (!result.success) {
      const errorMessage = result.error.errors.map(err => err.message).join(", ");
      return NextResponse.json(
        { success: false, message: errorMessage },
        { status: 400 }
      );
    }

    const { filename, contentType, fileSize } = result.data;

    const fileExtension = filename.split(".").pop();
    const uniqueKey = `users/${currentUser.id}/profile/${Date.now()}.${fileExtension}`;

    const putObjectCommand = new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: uniqueKey,
      ContentType: contentType,
      Metadata: {
        userId: currentUser.id,
      },
      ContentLength: fileSize,
    });

    const url = await getSignedUrl(R2, putObjectCommand, { expiresIn: 3600 });
    const imageUrl = `${process.env.R2_PUBLIC_DOMAIN}/${uniqueKey}`;

    const response = {
      success: true,
      message: "Upload URL generated successfully",
      data: { url, imageUrl },
    };

    return NextResponse.json(response);
  } catch (error: unknown) {
    console.error("Failed to generate upload URL:", error);
    return NextResponse.json(
      { success: false, message: "Failed to generate upload URL" },
      { status: 500 }
    );
  }
}
