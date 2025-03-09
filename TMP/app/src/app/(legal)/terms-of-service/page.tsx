import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Metadata } from "next";
import TermsOfServiceContent from "./content.mdx";

export const metadata: Metadata = {
  title: "Terms of Service",
  description: "Essential guidelines and conditions for using movieai.dev",
};

export default function ContactPage() {
  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>Terms of Service</PageTitle>
          <PageDescription>
            Essential guidelines and conditions for using movieai.dev.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <TermsOfServiceContent />
        </PageBody>
      </PageLayout>
    </Container>
  );
}
