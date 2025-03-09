import Container from "@/components/container";
import ResetPasswordForm from "@/features/auth/components/reset-password-form";
import { Metadata } from "next";
import { notFound } from "next/navigation";

export const metadata: Metadata = {
  title: "Set New Password",
  description: "Create a new password for your MovieAI account",
};

interface ResetPasswordPageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

const ResetPasswordPage = async ({ searchParams }: ResetPasswordPageProps) => {
  const searchParamsResolved = await searchParams;
  const email = searchParamsResolved.email as string;
  const token = searchParamsResolved.token as string;

  if (!email || !token) {
    notFound();
  }

  return (
    <Container variant="center">
      <ResetPasswordForm email={email} token={token} />;
    </Container>
  );
};

export default ResetPasswordPage;
