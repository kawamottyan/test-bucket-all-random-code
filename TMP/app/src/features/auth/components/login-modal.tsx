"use client";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
import { Input } from "@/components/ui/input";
import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useAuthModalStore } from "@/stores/auth-modal-store";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, Eye, EyeOff } from "lucide-react";
import { signIn, useSession } from "next-auth/react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { SiGithub, SiGoogle } from "react-icons/si";
import { toast } from "sonner";
import { login } from "../actions/login";
import { LoginFormValues, LoginSchema } from "../schemas";

interface LoginModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function LoginModal({ open, onOpenChange }: LoginModalProps) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get("callbackUrl");

  const [showPassword, setShowPassword] = useState(false);
  const [isGoogleAuth, setIsGoogleAuth] = useState(false);
  const [isGithubAuth, setIsGithubAuth] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | undefined>("");
  const [isPending, startTransition] = useTransition();

  const { setRegisterModalOpen } = useAuthModalStore();
  const queryClient = useQueryClient();
  const { update: updateSession } = useSession();
  const router = useRouter();

  const form = useForm<LoginFormValues>({
    resolver: zodResolver(LoginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const onSubmit = (values: LoginFormValues) => {
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await login(values);
        if (response.success) {
          queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
          await updateSession();
          form.reset();
          onOpenChange(false);
          toast.success("Successfully logged in");
          if (callbackUrl) {
            router.push(callbackUrl);
          }
        } else {
          setErrorMessage(response.message);
        }
      } catch {
        setErrorMessage("Failed to login with email. Please try again.");
      }
    });
  };

  const handleSocialLogin = async (provider: "google" | "github") => {
    const setAuth = provider === "google" ? setIsGoogleAuth : setIsGithubAuth;
    setAuth(true);
    setErrorMessage("");

    try {
      await signIn(provider, {
        callbackUrl: callbackUrl || pathname,
      });
    } catch {
      setErrorMessage("Failed to login with social account. Please try again.");
    } finally {
      setAuth(false);
    }
  };


  const handleRegisterClick = () => {
    onOpenChange(false);
    setRegisterModalOpen(true);
  };

  const handlePasswordReset = () => {
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md" aria-modal="true">
        <DialogHeader>
          <DialogTitle className="md:text-2xl">Log in</DialogTitle>
          <DialogDescription>
            Log in faster with your favorite social account.
          </DialogDescription>
        </DialogHeader>

        {errorMessage && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}

        <div className="mt-4 grid grid-cols-2 gap-6">
          <Button
            variant="outline"
            onClick={() => handleSocialLogin("google")}
            disabled={isGoogleAuth || isGithubAuth}
            className="w-full"
          >
            <SiGoogle className="mr-2 h-4 w-4" />
            {isGoogleAuth ? "Loading..." : "Google"}
          </Button>
          <Button
            variant="outline"
            onClick={() => handleSocialLogin("github")}
            disabled={isGoogleAuth || isGithubAuth}
            className="w-full"
          >
            <SiGithub className="mr-2 h-4 w-4" />
            {isGithubAuth ? "Loading..." : "GitHub"}
          </Button>
        </div>

        <div className="relative my-4">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-background px-2 text-muted-foreground">
              Or continue with email
            </span>
          </div>
        </div>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <div className="space-y-4">
              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Name@example.com"
                        type="email"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="password"
                render={({ field }) => (
                  <FormItem>
                    <div className="flex items-center justify-between">
                      <FormLabel>Password</FormLabel>
                      <Link
                        href="/auth/initiate-password-reset"
                        className="text-xs underline"
                        onClick={handlePasswordReset}
                      >
                        Forgot password?
                      </Link>
                    </div>
                    <div className="relative">
                      <FormControl>
                        <Input
                          placeholder="Use letters, numbers, and symbols"
                          type={showPassword ? "text" : "password"}
                          {...field}
                        />
                      </FormControl>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPassword(!showPassword)}
                        tabIndex={-1}
                      >
                        {showPassword ? (
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
              {isPending ? "Logging in..." : "Login with Email"}
            </Button>
          </form>
        </Form>

        <DialogFooter className="block">
          <div className="flex flex-col items-center space-y-4">
            <p className="text-center text-xs text-muted-foreground">
              By continuing, you agree to our{" "}
              <a href="/terms-of-service" target="_blank" className="underline">
                Terms
              </a>{" "}
              and{" "}
              <a href="/privacy-policy" target="_blank" className="underline">
                Privacy
              </a>
              .
            </p>

            <p className="text-center text-sm text-muted-foreground">
              Don&apos;t have an account?{" "}
              <a
                onClick={handleRegisterClick}
                className="cursor-pointer text-primary"
              >
                Sign up
              </a>
            </p>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default LoginModal;
