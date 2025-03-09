import Container from "@/components/container";
import VerifyEmailForm from "@/features/auth/components/verify-email-form";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Verify Email Update",
  description: "Verify your email address update to continue using MovieAI",
};

interface VerifyEmailUpdatePageProps {
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

const VerifyEmailUpdatePage = async ({
  searchParams,
}: VerifyEmailUpdatePageProps) => {
  const email = (await searchParams)?.email as string;

  return (
    <Container variant="center">
      <VerifyEmailForm email={email} type="update" />
    </Container>
  );
};

export default VerifyEmailUpdatePage;
