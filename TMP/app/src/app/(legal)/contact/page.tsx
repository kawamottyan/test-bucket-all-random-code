import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Metadata } from "next";
import ContactContent from "./content.mdx";

export const metadata: Metadata = {
  title: "Contact",
  description: "Connect with me through email or social media platforms",
};

export default function ContactPage() {
  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>Contact</PageTitle>
          <PageDescription>
            Connect with me through email or social media platforms.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <ContactContent />
        </PageBody>
      </PageLayout>
    </Container>
  );
}
