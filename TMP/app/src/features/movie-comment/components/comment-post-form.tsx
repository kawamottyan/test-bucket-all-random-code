import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormMessage,
} from "@/components/ui/form";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { CommentFormSchema, CommentFormValues } from "../schemas";

interface CommentPostFormProps {
  defaultValues: CommentFormValues;
  setInput: (value: string) => void;
  isEditing: boolean;
  setIsEditing: (value: boolean) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (values: CommentFormValues) => void;
}

export function CommentPostForm({
  defaultValues,
  setInput,
  isEditing,
  setIsEditing,
  open,
  onOpenChange,
  onSubmit,
}: CommentPostFormProps) {
  const form = useForm<CommentFormValues>({
    resolver: zodResolver(CommentFormSchema),
    defaultValues,
    values: defaultValues,
  });

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Confirm Your Comment</AlertDialogTitle>
          <AlertDialogDescription>
            Please review your comment before posting.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="content"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    {isEditing ? (
                      <Textarea
                        {...field}
                        className="mt-4 bg-muted/50"
                        autoFocus
                        onChange={(e) => {
                          field.onChange(e);
                          setInput(e.target.value);
                        }}
                      />
                    ) : (
                      <ScrollArea className="max-h-1/2 mt-4">
                        <div
                          onClick={() => setIsEditing(true)}
                          className="break-words rounded-md border bg-muted/50 p-3 text-sm text-muted-foreground hover:bg-muted/70"
                        >
                          {field.value}
                        </div>
                      </ScrollArea>
                    )}
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="isSpoiler"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center space-x-2">
                  <FormControl>
                    <Checkbox
                      id="isSpoiler"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <Label htmlFor="isSpoiler" className="pb-2 leading-none">
                    This comment contains plot details
                  </Label>
                </FormItem>
              )}
            />

            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <Button type="submit">Post</Button>
            </AlertDialogFooter>
          </form>
        </Form>
      </AlertDialogContent>
    </AlertDialog>
  );
}
