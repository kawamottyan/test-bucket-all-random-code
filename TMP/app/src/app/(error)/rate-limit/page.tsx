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
import Link from "next/link";

export default function RatelimitPage() {
  return (
    <Container variant="center">
      <PageLayout>
        <PageHeader>
          <PageTitle>429</PageTitle>
          <PageDescription>
            Too many requests, please try again later.
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
