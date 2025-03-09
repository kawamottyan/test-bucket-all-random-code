"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
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
import { Input } from "@/components/ui/input";
import { zodResolver } from "@hookform/resolvers/zod";
import { AlertCircle } from "lucide-react";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { initiateEmailUpdate } from "../actions/update-email";
import { UpdateEmailFormValues, UpdateEmailSchema } from "../schemas";

interface EmailModalProps {
  email: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function EmailModal({
  email,
  open,
  onOpenChange,
}: EmailModalProps) {
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isPending, startTransition] = useTransition();

  const form = useForm<UpdateEmailFormValues>({
    resolver: zodResolver(UpdateEmailSchema),
    defaultValues: {
      email: "",
    },
  });

  const onSubmit = (values: UpdateEmailFormValues) => {
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await initiateEmailUpdate(values);

        if (response.success) {
          form.reset();
          onOpenChange(false);
          toast.success(
            "A verification code has been sent. Please check your inbox."
          );
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to update email. Please try again.");
      }
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md" aria-modal="true">
        <DialogHeader>
          <DialogTitle className="md:text-2xl">Update Email</DialogTitle>
          <DialogDescription>Enter your new email address.</DialogDescription>
        </DialogHeader>

        {errorMessage && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <div className="space-y-4">
              <div>
                <div className="text-sm font-medium">Current Email</div>
                <div className="mt-2 rounded-md border bg-muted/50 px-3 py-2">
                  <div className="text-base text-muted-foreground">{email}</div>
                </div>
              </div>
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>New Email</FormLabel>
                    <FormControl>
                      <Input
                        type="email"
                        placeholder="Enter new email"
                        disabled={isPending}
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <Button type="submit" className="mt-8 w-full" disabled={isPending}>
              {isPending ? "Updating Email..." : "Update Email"}
            </Button>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
}
