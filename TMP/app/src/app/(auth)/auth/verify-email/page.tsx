import Container from "@/components/container";
import VerifyEmailForm from "@/features/auth/components/verify-email-form";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Verify Email",
  description: "Verify your email address to access your MovieAI account",
};

interface VerifyEmailPageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

const VerifyEmailPage = async ({ searchParams }: VerifyEmailPageProps) => {
  const email = (await searchParams)?.email as string;

  return (
    <Container variant="center">
      <VerifyEmailForm email={email} type="verification" />
    </Container>
  );
};

export default VerifyEmailPage;
