import { SidebarProvider } from "@/components/ui/sidebar";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { AppSidebar } from "@/features/navigation/app-sidebar";
import Header from "@/features/navigation/header";
import { auth } from "@/lib/auth";
import { cn } from "@/lib/utils";
import { ModalProvider } from "@/providers/modal-provider";
import { QueryProvider } from "@/providers/query-provider";
import SessionInitializer from "@/providers/session-initializer";
import { ThemeProvider } from "@/providers/theme-provider";
import { TrainingConsentProvider } from "@/providers/training-consent-provider";
import UuidInitializer from "@/providers/uuid-initializer";
import type { Metadata } from "next";
import { SessionProvider } from "next-auth/react";
import { Inter as FontSans } from "next/font/google";
import { cookies } from "next/headers";
import "./globals.css";

const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: {
    default: "MovieAI",
    template: "%s - MovieAI",
  },
  description:
    "Discover your next favorite movie with AI-powered recommendations!",
  twitter: {
    card: "summary_large_image",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const session = await auth();
  const cookieStore = await cookies();
  const defaultOpen = cookieStore.get("sidebar:state")?.value === "true";

  return (
    <SessionProvider session={session}>
      <QueryProvider>
        <html lang="en" suppressHydrationWarning>
          <body
            className={cn(
              "bg-background",
              "font-sans",
              "antialiased",
              fontSans.variable
            )}
          >
            <ThemeProvider
              attribute="class"
              defaultTheme="dark"
              enableSystem
              disableTransitionOnChange
            >
              <UuidInitializer />
              <SessionInitializer />
              <ModalProvider />
              <TrainingConsentProvider />
              <SidebarProvider defaultOpen={defaultOpen}>
                <AppSidebar />
                <Header />
                <main className="min-h-dvh w-full">
                  <Sonner />
                  {children}
                </main>
              </SidebarProvider>
            </ThemeProvider>
          </body>
        </html>
      </QueryProvider>
    </SessionProvider>
  );
}
