import { z } from "zod";

const isValidDate = (dateStr: string): boolean => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
    return false;
  }

  const [year, month, day] = dateStr.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  return (
    date.getFullYear() === year &&
    date.getMonth() === month - 1 &&
    date.getDate() === day
  );
};

export const ReviewFormSchema = z.object({
  watchDate: z
    .string()
    .min(1, "Watch date is required")
    .refine(
      (date) => isValidDate(date),
      () => ({
        message: "Please enter a real date in YYYY-MM-DD format",
      })
    ),
  rating: z
    .number()
    .gt(0, "Rating must be greater than 0")
    .max(5, "Rating must be 5 or less"),
  note: z.string().optional(),
});

export const ReviewApiSchema = ReviewFormSchema.extend({
  movieId: z.number(),
});

export type ReviewFormValues = z.infer<typeof ReviewFormSchema>;
export type ReviewApiValues = z.infer<typeof ReviewApiSchema>;
