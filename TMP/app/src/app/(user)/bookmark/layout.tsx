import { Metadata } from "next";

export const metadata: Metadata = {
  title: "My Bookmarks",
  description: "Access and manage your saved movies and bookmarked content",
};

export default function ReviewLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
