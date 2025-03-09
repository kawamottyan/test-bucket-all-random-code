"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import Link from "next/link";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { initiatePasswordReset } from "../actions/password";
import {
  InitiatePasswordResetFormValues,
  InitiatePasswordResetSchema,
} from "../schemas";

export default function InitiatePasswordResetForm() {
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isSuccess, setIsSuccess] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [submittedEmail, setSubmittedEmail] = useState<string>("");

  const form = useForm<InitiatePasswordResetFormValues>({
    resolver: zodResolver(InitiatePasswordResetSchema),
    defaultValues: {
      email: "",
    },
  });

  const onSubmit = (values: InitiatePasswordResetFormValues) => {
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await initiatePasswordReset(values);
        if (response.success) {
          setSubmittedEmail(values.email);
          form.reset();
          setIsSuccess(true);
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to send reset email. Please try again.");
      }
    });
  };

  if (isSuccess) {
    return (
      <Card className="mx-4 mx-auto w-full max-w-md">
        <CardHeader>
          <CardTitle>Check Your Email</CardTitle>
          <CardDescription>
            If an account exists for {submittedEmail}, you will receive a
            password reset link.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Link href="/">
            <Button className="w-full">Go to Home</Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="mx-4 mx-auto w-full max-w-md">
      <CardHeader>
        <CardTitle>Reset Password</CardTitle>
        <CardDescription>
          Enter your email address and we&apos;ll send you a link to reset your
          password.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {errorMessage && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <div className="space-y-4">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email Address</FormLabel>
                    <FormControl>
                      <Input
                        type="email"
                        placeholder="Enter your email address"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <Button type="submit" className="mt-8 w-full" disabled={isPending}>
              {isPending ? "Sending Reset Link..." : "Send Reset Link"}
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
