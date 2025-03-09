import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Login",
  description:
    "Log in to your MovieAI account to access personalized movie recommendations and features",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
