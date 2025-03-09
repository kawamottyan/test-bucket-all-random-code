"use client";

import Container from "@/components/container";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
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
import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { login } from "@/features/auth/actions/login";
import { getAuthMessage } from "@/features/auth/lib/messages";
import { LoginFormValues, LoginSchema } from "@/features/auth/schemas";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle, CheckCircle, Eye, EyeOff } from "lucide-react";
import { signIn, useSession } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, useTransition } from "react";
import { useForm } from "react-hook-form";
import { SiGithub, SiGoogle } from "react-icons/si";
import { toast } from "sonner";

const LoginPage = () => {
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get("callbackUrl");
  const [messageCode, setMessageCode] = useState(
    searchParams.get("status") || searchParams.get("error")
  );
  const [errorMessage, setErrorMessage] = useState<string | undefined>("");
  const [showPassword, setShowPassword] = useState(false);
  const [isGoogleAuth, setIsGoogleAuth] = useState(false);
  const [isGithubAuth, setIsGithubAuth] = useState(false);
  const [isPending, startTransition] = useTransition();

  const message = getAuthMessage(messageCode);

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
    setMessageCode(null);
    setErrorMessage("");

    startTransition(async () => {
      try {
        const response = await login(values);
        if (response.success) {
          queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
          await updateSession();
          form.reset();
          router.push(callbackUrl || "/");
          toast.success("Successfully logged in");
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
    setMessageCode(null);
    setErrorMessage("");

    try {
      const result = await signIn(provider, {
        callbackUrl: callbackUrl || "/",
      });

      if (result?.error) {
        setMessageCode("SocialSignInError");
      }
    } catch {
      setErrorMessage("Failed to login with social account. Please try again.");
    } finally {
      setAuth(false);
    }
  };

  return (
    <Container variant="center">
      <Card className="mx-4 mx-auto w-full max-w-md">
        <CardHeader>
          <CardTitle>Log in</CardTitle>
          <CardDescription>
            Log in faster with your favorite social account.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {errorMessage ? (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{errorMessage}</AlertDescription>
            </Alert>
          ) : message && (
            <Alert variant={message.variant} className="mb-6">
              {message.variant === "default" ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
              <AlertTitle>{message.title}</AlertTitle>
              <AlertDescription>{message.description}</AlertDescription>
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
                      <FormLabel>Password</FormLabel>
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

              <Button
                type="submit"
                className="mt-8 w-full"
                disabled={isPending}
              >
                {isPending ? "Logging in..." : "Login with Email"}
              </Button>
            </form>
          </Form>
        </CardContent>
        <CardFooter className="block">
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
          </div>
        </CardFooter>
      </Card>
    </Container>
  );
};

export default LoginPage;
