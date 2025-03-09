import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Authentication Error",
  description: "There was a problem with your authentication request",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
