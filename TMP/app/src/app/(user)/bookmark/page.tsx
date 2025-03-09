"use client";

import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import BookmarkCard from "@/features/movie-bookmark/components/bookmark-card";
import { BookmarkControls } from "@/features/movie-bookmark/components/bookmark-controls";
import { BookmarkLoading } from "@/features/movie-bookmark/components/bookmark-skeleton";
import { useFilteredBookmarks } from "@/features/movie-bookmark/hooks/use-filtered-bookmarks";

const BookmarkPage = () => {
  const { bookmarks, isPending, search, setSearch, sortBy, setSortBy } =
    useFilteredBookmarks();

  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>My Bookmarks</PageTitle>
          <PageDescription>
            Access your saved movies and manage your bookmarked content.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <BookmarkControls
            search={search}
            onSearchChange={setSearch}
            sortBy={sortBy}
            onSortChange={setSortBy}
          />

          {isPending ? (
            <BookmarkLoading />
          ) : bookmarks.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="text-lg font-medium">No bookmarks found</div>
              <div className="mt-2 text-sm text-muted-foreground">
                {search
                  ? "Try adjusting your search or filter"
                  : "Start adding some bookmarks to your collection"}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {bookmarks.map((bookmark) => (
                <BookmarkCard key={bookmark.id} bookmark={bookmark} />
              ))}
            </div>
          )}
        </PageBody>
      </PageLayout>
    </Container>
  );
};

export default BookmarkPage;
