"use client";

import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { formatIsoDate, formatIsoDateInput } from "@/lib/formatter";
import { zodResolver } from "@hookform/resolvers/zod";
import { CalendarIcon } from "lucide-react";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { useReviewActions } from "../hooks/use-review-actions";
import { ReviewFormSchema, ReviewFormValues } from "../schemas";

interface ModalAdvancedReviewProps {
  movieId: number;
  onOpenChange: (open: boolean) => void;
}

export function ModalAdvancedReview({
  movieId,
  onOpenChange,
}: ModalAdvancedReviewProps) {
  const { review, isReviewed, isPending, addReview } =
    useReviewActions(movieId);

  const form = useForm<ReviewFormValues>({
    resolver: zodResolver(ReviewFormSchema),
    defaultValues: {
      watchDate: formatIsoDate(new Date()),
      rating: 0,
      note: "",
    },
  });

  useEffect(() => {
    if (isReviewed && review) {
      form.reset({
        watchDate: formatIsoDate(new Date(review.watchedAt)),
        rating: review.rating,
        note: review.note || "",
      });
    } else {
      form.reset({
        watchDate: formatIsoDate(new Date()),
        rating: 0,
        note: "",
      });
    }
  }, [isReviewed, review, form]);

  const handleSubmit = (values: ReviewFormValues) => {
    const date = new Date(values.watchDate);
    if (date instanceof Date && !isNaN(date.getTime())) {
      addReview({
        rating: values.rating,
        watchDate: values.watchDate,
        note: values.note,
      });
      onOpenChange(false);
    }
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(handleSubmit)}
        className="space-y-8 pt-2"
      >
        <FormField
          control={form.control}
          name="watchDate"
          render={({ field }) => (
            <FormItem className="space-y-2">
              <FormLabel>Watch Date</FormLabel>
              <FormControl>
                <div className="flex gap-2">
                  <Popover modal={true}>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        size="icon"
                        className="shrink-0"
                        type="button"
                      >
                        <CalendarIcon className="h-4 w-4" />
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={
                          field.value ? new Date(field.value) : undefined
                        }
                        onSelect={(date) => {
                          if (date) {
                            field.onChange(formatIsoDate(date));
                          }
                        }}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <Input
                    type="text"
                    {...field}
                    onChange={(e) => field.onChange(e.target.value)}
                    onBlur={(e) => {
                      const formatted = formatIsoDateInput(e.target.value);
                      field.onChange(formatted);
                    }}
                    placeholder="YYYY-MM-DD"
                    className="w-full"
                  />
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="rating"
          render={({ field }) => (
            <FormItem className="space-y-1">
              <FormLabel>Rating</FormLabel>
              <FormControl>
                <div className="flex items-center space-x-4">
                  <Slider
                    min={0}
                    max={5}
                    step={0.1}
                    value={[field.value]}
                    onValueChange={(value) => field.onChange(value[0])}
                    className="w-full"
                  />
                  <Input
                    type="number"
                    value={field.value}
                    onChange={(e) => field.onChange(Number(e.target.value))}
                    min={0}
                    max={5}
                    step={0.1}
                    className="w-20"
                  />
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="note"
          render={({ field }) => (
            <FormItem className="space-y-2">
              <FormLabel>Review Notes</FormLabel>
              <FormControl>
                <Textarea
                  {...field}
                  placeholder="Write your thoughts about the movie..."
                  className="min-h-20"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" className="w-full" disabled={isPending}>
          {isReviewed ? "Update Review" : "Submit Review"}
        </Button>
      </form>
    </Form>
  );
}
