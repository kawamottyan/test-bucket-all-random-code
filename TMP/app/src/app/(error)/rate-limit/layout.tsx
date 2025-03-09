import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Rate Limit",
  description: "Too many requests, please try again later",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
