import { Metadata } from "next";

export const metadata: Metadata = {
  title: "My Reviews",
  description: "View and manage all your movie reviews in one place",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
