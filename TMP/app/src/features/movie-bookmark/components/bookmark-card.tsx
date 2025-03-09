import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { MovieReviewModal } from "@/features/movie-review/components/movie-review-modal";
import { useReviewActions } from "@/features/movie-review/hooks/use-review-actions";
import { Bookmark, Star } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { useBookmarkActions } from "../hooks/use-bookmark-actions";
import { SafeBookmark } from "../types";

interface BookmarkCardProps {
  bookmark: SafeBookmark;
}

const BookmarkCard = ({ bookmark }: BookmarkCardProps) => {
  const { movie } = bookmark;
  const [isReviewModalOpen, setReviewModalOpen] = useState(false);

  const {
    isBookmarked,
    isPending: isBookmarkPending,
    toggleBookmark,
  } = useBookmarkActions(movie.movieId);

  const { isReviewed, isPending: isReviewPending } = useReviewActions(
    movie.movieId
  );

  const handleBookmarkClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();

    toggleBookmark();
  };

  const handleReviewClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setReviewModalOpen(true);
  };

  return (
    <Link href={`/movies/${movie.movieId}`}>
      <Card className="h-48 p-0">
        <CardContent className="h-full p-0">
          <div className="flex h-full">
            <div className="relative w-32">
              <Image
                src={movie.posterPath}
                alt={movie.title}
                fill
                className="rounded-l-md object-cover"
                priority
              />
            </div>
            <div className="flex-1 p-4">
              <div className="flex h-full flex-col justify-between">
                <div className="flex flex-col gap-y-1">
                  <div className="line-clamp-1">{movie.title}</div>
                  <p className="line-clamp-3 text-sm text-muted-foreground">
                    {movie.overview}
                  </p>
                </div>
                <div className="flex gap-x-2">
                  <Button
                    variant="secondary"
                    size="icon"
                    onClick={handleBookmarkClick}
                    className="h-10 w-10 rounded-full"
                    disabled={isBookmarkPending}
                  >
                    <Bookmark
                      size={32}
                      className={isBookmarked ? "fill-foreground" : "fill-none"}
                    />
                  </Button>
                  <Button
                    variant="secondary"
                    size="icon"
                    onClick={handleReviewClick}
                    className="h-10 w-10 rounded-full"
                    disabled={isReviewPending}
                  >
                    <Star
                      size={32}
                      className={isReviewed ? "fill-foreground" : "fill-none"}
                    />
                  </Button>
                  <MovieReviewModal
                    movieId={movie.movieId}
                    open={isReviewModalOpen}
                    onOpenChange={setReviewModalOpen}
                  />
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
};

export default BookmarkCard;
