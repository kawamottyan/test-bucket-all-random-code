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
import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import {
  setInitialPassword,
  updatePassword,
} from "@/features/auth/actions/password";
import {
  SetPasswordFormValues,
  SetPasswordSchema,
} from "@/features/auth/schemas";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Eye, EyeOff } from "lucide-react";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { toast } from "sonner";
import { UpdatePasswordFormValues, UpdatePasswordSchema } from "../schemas";

interface PasswordModalProps {
  hasPassword: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function PasswordModal({
  hasPassword,
  open,
  onOpenChange,
}: PasswordModalProps) {
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmNewPassword, setShowConfirmNewPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isPending, startTransition] = useTransition();
  const queryClient = useQueryClient();

  const updatePasswordForm = useForm<UpdatePasswordFormValues>({
    resolver: zodResolver(UpdatePasswordSchema),
    defaultValues: {
      currentPassword: "",
      newPassword: "",
      confirmNewPassword: "",
    },
  });

  const setPasswordForm = useForm<SetPasswordFormValues>({
    resolver: zodResolver(SetPasswordSchema),
    defaultValues: {
      password: "",
      confirmPassword: "",
    },
  });

  const handleUpdatePassword = (values: UpdatePasswordFormValues) => {
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await updatePassword(values);

        if (response.success) {
          updatePasswordForm.reset();
          queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
          onOpenChange(false);
          toast.success("Your password has been updated successfully");
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to update password. Please try again.");
      }
    });
  };

  const handleSetPassword = (values: SetPasswordFormValues) => {
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await setInitialPassword(values);

        if (response.success) {
          setPasswordForm.reset();
          queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
          onOpenChange(false);
          toast.success("Your password has been set successfully");
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to set password. Please try again.");
      }
    });
  };

  const renderUpdatePasswordForm = () => (
    <Form {...updatePasswordForm}>
      <form onSubmit={updatePasswordForm.handleSubmit(handleUpdatePassword)}>
        <div className="space-y-4">
          <FormField
            control={updatePasswordForm.control}
            name="currentPassword"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Current Password</FormLabel>
                <div className="relative">
                  <FormControl>
                    <Input
                      type={showCurrentPassword ? "text" : "password"}
                      placeholder="Enter your current password"
                      {...field}
                    />
                  </FormControl>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() =>
                      setShowCurrentPassword(!showCurrentPassword)
                    }
                    tabIndex={-1}
                  >
                    {showCurrentPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={updatePasswordForm.control}
            name="newPassword"
            render={({ field }) => (
              <FormItem>
                <FormLabel>New Password</FormLabel>
                <div className="relative">
                  <FormControl>
                    <Input
                      type={showNewPassword ? "text" : "password"}
                      placeholder="Use letters, numbers, and symbols"
                      {...field}
                    />
                  </FormControl>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    tabIndex={-1}
                  >
                    {showNewPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={updatePasswordForm.control}
            name="confirmNewPassword"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Confirm New Password</FormLabel>
                <div className="relative">
                  <FormControl>
                    <Input
                      type={showConfirmNewPassword ? "text" : "password"}
                      placeholder="Confirm your new password"
                      {...field}
                    />
                  </FormControl>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() =>
                      setShowConfirmNewPassword(!showConfirmNewPassword)
                    }
                    tabIndex={-1}
                  >
                    {showConfirmNewPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <Button type="submit" className="mt-8 w-full" disabled={isPending}>
          {isPending ? "Updating Password..." : "Update Password"}
        </Button>
      </form>
    </Form>
  );

  const renderSetPasswordForm = () => (
    <Form {...setPasswordForm}>
      <form onSubmit={setPasswordForm.handleSubmit(handleSetPassword)}>
        <div className="space-y-4">
          <FormField
            control={setPasswordForm.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Password</FormLabel>
                <div className="relative">
                  <FormControl>
                    <Input
                      type={showNewPassword ? "text" : "password"}
                      placeholder="Use letters, numbers, and symbols"
                      {...field}
                    />
                  </FormControl>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    tabIndex={-1}
                  >
                    {showNewPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
          <FormField
            control={setPasswordForm.control}
            name="confirmPassword"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Confirm Password</FormLabel>
                <div className="relative">
                  <FormControl>
                    <Input
                      type={showConfirmNewPassword ? "text" : "password"}
                      placeholder="Confirm your password"
                      {...field}
                    />
                  </FormControl>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() =>
                      setShowConfirmNewPassword(!showConfirmNewPassword)
                    }
                    tabIndex={-1}
                  >
                    {showConfirmNewPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>
        <Button type="submit" className="mt-8 w-full" disabled={isPending}>
          {isPending ? "Setting Password..." : "Set Password"}
        </Button>
      </form>
    </Form>
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md" aria-modal="true">
        <DialogHeader>
          <DialogTitle className="text-2xl">
            {hasPassword ? "Update Password" : "Set Password"}
          </DialogTitle>
          <DialogDescription>
            {hasPassword
              ? "Change your account password."
              : "Set up your account password."}
          </DialogDescription>
        </DialogHeader>

        {errorMessage && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}

        {hasPassword ? renderUpdatePasswordForm() : renderSetPasswordForm()}
      </DialogContent>
    </Dialog>
  );
}