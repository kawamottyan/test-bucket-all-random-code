"use client";

import Container from "@/components/container";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { getAuthMessage } from "@/features/auth/lib/messages";
import { useAuthModalStore } from "@/stores/auth-modal-store";
import { AlertCircle } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

const ErrorPage = () => {
  const searchParams = useSearchParams();
  const messageCode = searchParams.get("error")
  const message = getAuthMessage(messageCode);
  const { setLoginModalOpen } = useAuthModalStore();

  return (
    <Container variant="center">
      <Card className="mx-auto w-full max-w-md">
        <CardHeader>
          <CardTitle>Authentication Error</CardTitle>
          <CardDescription>
            There was a problem with your authentication request.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {message && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>{message.title}</AlertTitle>
              <AlertDescription>{message.description}</AlertDescription>
            </Alert>
          )}
          <div className="flex flex-col gap-y-2">
            <Button onClick={() => setLoginModalOpen(true)} className="w-full">
              Log in
            </Button>
            <Link href="/" className="w-full">
              <Button variant="outline" className="w-full">
                Go to Home
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </Container>
  );
};

export default ErrorPage;
