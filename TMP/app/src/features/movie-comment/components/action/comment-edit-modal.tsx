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
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useCommentActions } from "../../hooks/use-comment-actions";
import { CommentEditFormSchema, CommentEditFormValues } from "../../schemas";

interface CommentEditModalProps {
  movieId: number;
  commentId: string;
  content: string;
  isSpoiler: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess: () => void;
}

export function CommentEditModal({
  movieId,
  commentId,
  content,
  isSpoiler,
  open,
  onOpenChange,
  onSuccess,
}: CommentEditModalProps) {
  const { editComment, isPending } = useCommentActions(movieId);

  const form = useForm<CommentEditFormValues>({
    resolver: zodResolver(CommentEditFormSchema),
    defaultValues: {
      content: content,
      isSpoiler: isSpoiler,
    },
  });

  const onSubmit = (values: CommentEditFormValues) => {
    editComment(
      {
        ...values,
        movieId,
        commentId,
      },
      {
        onSuccess: () => {
          onSuccess();
        },
      }
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <DialogHeader>
              <DialogTitle>Edit Comment</DialogTitle>
              <DialogDescription>
                Make changes to your comment.
              </DialogDescription>
            </DialogHeader>

            <FormField
              control={form.control}
              name="content"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Comment</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Write your comment here..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="isSpoiler"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Spoiler Warning</FormLabel>
                  <FormControl>
                    <Select
                      value={field.value ? "true" : "false"}
                      onValueChange={(value) =>
                        field.onChange(value === "true")
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select spoiler status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="false">No spoilers</SelectItem>
                        <SelectItem value="true">Contains spoilers</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter className="gap-2 pt-4">
              <Button
                type="reset"
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
