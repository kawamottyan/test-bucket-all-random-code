import { CommentReportType } from "@prisma/client";
import { z } from "zod";
import { MAX_COMMENT_LENGTH } from "./constants";

export const CommentSchema = z
  .string()
  .min(1, "Comment cannot be empty")
  .max(
    MAX_COMMENT_LENGTH,
    `Comment cannot exceed ${MAX_COMMENT_LENGTH} characters`
  );

export const CommentFormSchema = z.object({
  content: CommentSchema,
  isSpoiler: z.boolean(),
});

export const CommentApiSchema = CommentFormSchema.extend({
  movieId: z.number(),
  depth: z.number(),
  parentId: z.string().nullable(),
});

export const CommentReportFormSchema = z.object({
  type: z.nativeEnum(CommentReportType, {
    required_error: "You must select a report type",
  }),
  description: z.string().optional(),
});

export const CommentReportApiSchema = CommentReportFormSchema.extend({
  movieId: z.number(),
  commentId: z.string(),
});

export const CommentEditFormSchema = z.object({
  content: CommentSchema,
  isSpoiler: z.boolean().default(false),
});

export const CommentEditApiSchema = CommentEditFormSchema.extend({
  movieId: z.number(),
  commentId: z.string(),
});

export type CommentFormValues = z.infer<typeof CommentFormSchema>;
export type CommentApiValues = z.infer<typeof CommentApiSchema>;
export type CommentReportFormValues = z.infer<typeof CommentReportFormSchema>;
export type CommentReportApiValues = z.infer<typeof CommentReportApiSchema>;
export type CommentEditFormValues = z.infer<typeof CommentEditFormSchema>;
export type CommentEditApiValues = z.infer<typeof CommentEditApiSchema>;
