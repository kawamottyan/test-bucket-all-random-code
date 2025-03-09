import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Settings",
  description: "Manage your account information and customize your experience",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
