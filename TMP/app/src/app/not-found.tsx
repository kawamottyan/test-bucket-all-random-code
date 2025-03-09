"use client";

import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";
import { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Not Found",
  description: "Sorry, we couldn't find the page you're looking for",
};

export default function NotFound() {
  return (
    <Container variant="center">
      <PageLayout>
        <PageHeader>
          <PageTitle>404</PageTitle>
          <PageDescription>
            Sorry, we couldn&apos;t find the page you&apos;re looking for.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <Link href="/">
            <Button className="w-full">
              <ChevronLeft className="h-4 w-4" />
              Back to Home
            </Button>
          </Link>
        </PageBody>
      </PageLayout>
    </Container>
  );
}
