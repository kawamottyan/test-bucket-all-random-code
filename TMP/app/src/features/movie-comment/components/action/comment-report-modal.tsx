"use client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ServerResponse } from "@/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { useTransition } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import {
  CommentReportFormSchema,
  CommentReportFormValues,
} from "../../schemas";

interface CommentReportModalProps {
  movieId: number;
  commentId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

export function CommentReportModal({
  movieId,
  commentId,
  open,
  onOpenChange,
  onSuccess,
}: CommentReportModalProps) {
  const [isPending, startTransition] = useTransition();

  const form = useForm<CommentReportFormValues>({
    resolver: zodResolver(CommentReportFormSchema),
    defaultValues: {
      type: undefined,
      description: "",
    },
  });

  const onSubmit = async (values: CommentReportFormValues) => {
    startTransition(async () => {
      try {
        const response = await fetch("/api/comment/report", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            data: {
              ...values,
              movieId,
              commentId,
            },
          }),
        });

        const data: ServerResponse = await response.json();

        if (!response.ok) {
          toast.error(data.message || "Failed to submit report");
          return;
        }

        form.reset();
        onSuccess?.();
        onOpenChange(false);
        toast.success("Report submitted successfully");
      } catch (error) {
        console.error("Failed to submit report:", error);
        toast.error("An unexpected error occurred. Please try again.");
      }
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <DialogHeader>
              <DialogTitle>Report Issue</DialogTitle>
              <DialogDescription>
                Please provide the details below
              </DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Type</FormLabel>
                  <FormControl>
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select a report type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="SPOILER">Spoiler</SelectItem>
                        <SelectItem value="SPAM">Spam</SelectItem>
                        <SelectItem value="INAPPROPRIATE">
                          Inappropriate
                        </SelectItem>
                        <SelectItem value="OTHER">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Enter detailed description"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter className="gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => form.reset()}
                disabled={isPending}
              >
                Reset
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? "Submitting..." : "Submit"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
