import { FeedbackType } from "@prisma/client";
import { z } from "zod";

export const FeedbackFormSchema = z
  .object({
    type: z.nativeEnum(FeedbackType, {
      required_error: "Please select a feedback type",
    }),
    title: z.string().min(1, { message: "Title is required" }),
    description: z.string().min(1, { message: "Description is required" }),
    email: z.string().email("Invalid email").optional(),
  })
  .refine(
    (data) => {
      if (data.type === "SUPPORT") {
        return !!data.email;
      }
      return true;
    },
    {
      message: "Email is required for support requests",
      path: ["email"],
    }
  );

export type FeedbackFormValues = z.infer<typeof FeedbackFormSchema>;
