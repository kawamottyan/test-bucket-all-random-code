"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
  InputOTP,
  InputOTPGroup,
  InputOTPSlot,
} from "@/components/ui/input-otp";
import { resendVerificationToken } from "@/features/auth/actions/register";
import { OTPFormValues, OTPSchema } from "@/features/auth/schemas";
import { verifyEmailUpdate } from "@/features/setting/actions/update-email";
import { zodResolver } from "@hookform/resolvers/zod";
import { REGEXP_ONLY_DIGITS } from "input-otp";
import { AlertCircle, CheckCircle } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { verifyEmail } from "../actions/verify";

interface VerifyEmailFormProps {
  email: string | null;
  type: "verification" | "update";
}

const VerifyEmailForm = ({
  email: initialEmail,
  type,
}: VerifyEmailFormProps) => {
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [successMessage, setSuccessMessage] = useState<string>("");
  const [isPending, startTransition] = useTransition();
  const [isResending, startResending] = useTransition();
  const router = useRouter();

  const form = useForm<OTPFormValues>({
    resolver: zodResolver(OTPSchema),
    defaultValues: {
      email: initialEmail || "",
      otp: "",
    },
  });

  const onVerifyOTP = async (values: OTPFormValues) => {
    setErrorMessage("");
    setSuccessMessage("");

    startTransition(async () => {
      try {
        const response =
          type === "update"
            ? await verifyEmailUpdate(values)
            : await verifyEmail(values);

        if (response.success) {
          form.reset();
          if (type === "verification") {
            router.push("/auth/login?status=emailVerified");
          } else {
            setSuccessMessage("Email has been updated successfully.");
          }
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Invalid verification code");
      }
    });
  };

  const onResendOTP = async () => {
    setErrorMessage("");
    setSuccessMessage("");

    const email = form.getValues("email");

    startResending(async () => {
      try {
        const response = await resendVerificationToken({ email });
        if (response.success) {
          setSuccessMessage("Verification code has been resent to your email.");
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to resend verification code");
      }
    });
  };

  return (
    <Card className="mx-4 mx-auto w-full max-w-md">
      <CardHeader>
        <CardTitle>Verify Your Email</CardTitle>
      </CardHeader>
      <CardContent>
        {successMessage && (
          <Alert className="mb-6">
            <CheckCircle className="h-4 w-4" />
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>{successMessage}</AlertDescription>
          </Alert>
        )}

        {errorMessage && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onVerifyOTP)}>
            <div className="space-y-4">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>
                      {type === "update" ? "New Email" : "Email"}
                    </FormLabel>
                    <FormControl>
                      <Input
                        {...field}
                        type="email"
                        placeholder={
                          type === "update" ? "Enter New Email" : "Enter Email"
                        }
                        disabled={!!initialEmail}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="otp"
                render={({ field }) => (
                  <FormItem className="space-y-8">
                    <FormLabel>Verification Code</FormLabel>
                    <div className="flex justify-center">
                      <FormControl>
                        <InputOTP
                          maxLength={6}
                          pattern={REGEXP_ONLY_DIGITS}
                          {...field}
                        >
                          <InputOTPGroup>
                            <InputOTPSlot index={0} />
                            <InputOTPSlot index={1} />
                            <InputOTPSlot index={2} />
                            <InputOTPSlot index={3} />
                            <InputOTPSlot index={4} />
                            <InputOTPSlot index={5} />
                          </InputOTPGroup>
                        </InputOTP>
                      </FormControl>
                    </div>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <Button type="submit" className="mt-8 w-full" disabled={isPending}>
              {isPending ? "Verifying..." : "Verify Email"}
            </Button>

            {type !== "update" && (
              <Button
                type="button"
                variant="outline"
                className="mt-4 w-full"
                onClick={onResendOTP}
                disabled={isResending}
              >
                {isResending ? "Sending..." : "Resend Code"}
              </Button>
            )}
          </form>
        </Form>
      </CardContent>
    </Card>
  );
};

export default VerifyEmailForm;
