"use client";

import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { ReviewTable } from "@/features/movie-review/components/review-table";
import { useFetchReviews } from "@/features/movie-review/hooks/use-fetch-reviews";

const ReviewPage = () => {
  const { data: reviews = [], isPending } = useFetchReviews();

  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>My Reviews</PageTitle>
          <PageDescription>
            View and manage all your movie reviews in one place.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <ReviewTable data={reviews} isPending={isPending} />
        </PageBody>
      </PageLayout>
    </Container>
  );
};

export default ReviewPage;
