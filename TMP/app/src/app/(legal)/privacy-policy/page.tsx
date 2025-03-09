import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Metadata } from "next";
import PrivacyPolicyContent from "./content.mdx";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "Learn how we collect, use, and protect your personal information",
};

export default function ContactPage() {
  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>Privacy Policy</PageTitle>
          <PageDescription>
            Learn how we collect, use, and protect your personal information.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <PrivacyPolicyContent />
        </PageBody>
      </PageLayout>
    </Container>
  );
}
