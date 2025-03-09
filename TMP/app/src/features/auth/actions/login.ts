"use server";

import { signIn } from "@/lib/auth";
import { db } from "@/lib/db";
import { sendVerificationEmail } from "@/lib/email";
import { generateToken } from "@/lib/tokens";
import { ServerResponse } from "@/types/index";
import bcrypt from "bcryptjs";
import { AuthError } from "next-auth";
import { getUserByEmail } from "../../../actions/get-user";
import { EXPIRATION_TIMEOUT, OTP_TOKEN_LENGTH } from "../constants";
import { LoginFormValues, LoginSchema } from "../schemas";

export const login = async (
  values: LoginFormValues
): Promise<ServerResponse> => {
  const validatedFields = LoginSchema.safeParse(values);

  if (!validatedFields.success) {
    return { success: false, message: "Invalid request" };
  }

  const { email, password } = validatedFields.data;
  const existingUser = await getUserByEmail(email);

  if (!existingUser || !existingUser.email) {
    return { success: false, message: "Email does not exist" };
  }

  if (!existingUser.password) {
    return { success: false, message: "Password does not exist" };
  }

  if (!existingUser.emailVerified) {
    const token = await generateToken({
      length: OTP_TOKEN_LENGTH,
      type: "numeric",
    });
    const expires = new Date(new Date().getTime() + EXPIRATION_TIMEOUT);

    await db.emailVerificationToken.create({
      data: {
        email,
        token,
        expires,
      },
    });

    await sendVerificationEmail(email, token, "verification");
    return {
      success: false,
      message:
        "Please verify your email. A new verification code has been sent.",
    };
  }

  const isValidPassword = await bcrypt.compare(password, existingUser.password);
  if (!isValidPassword) {
    return { success: false, message: "Invalid password" };
  }

  try {
    await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
  } catch (error) {
    console.error("Failed to login with email:", error);
    if (error instanceof AuthError) {
      switch (error.type) {
        case "CredentialsSignin":
          return { success: false, message: "Invalid credentials" };
        default:
          return { success: false, message: "Something went wrong" };
      }
    }

    throw error;
  }
  return { success: true, message: "Successfully logged in" };
};