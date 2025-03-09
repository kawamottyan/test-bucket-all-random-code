import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Metadata } from "next";
import CommunityContent from "./content.mdx";

export const metadata: Metadata = {
  title: "Community",
  description: "Join our community where we share knowledge",
};

export default function CommunityPage() {
  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>Community</PageTitle>
          <PageDescription>
            Join our community where we share knowledge.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <CommunityContent />
        </PageBody>
      </PageLayout>
    </Container>
  );
}
