"use client";

import AuthButton from "@/components/auth-button";
import { Button } from "@/components/ui/button";
import { useBookmarkActions } from "@/features/movie-bookmark/hooks/use-bookmark-actions";
import CommentDrawer from "@/features/movie-comment/components/comment-drawer";
import { SafeComment } from "@/features/movie-comment/types";
import { MovieReviewModal } from "@/features/movie-review/components/movie-review-modal";
import { useReviewActions } from "@/features/movie-review/hooks/use-review-actions";
import { useAuthModalStore } from "@/stores/auth-modal-store";
import { useReplyStore } from "@/stores/use-reply-store";
import { Bookmark, ChevronLeft, MessageSquare, Play, Star } from "lucide-react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SafeMovie } from "../types";
import { MovieInfoModal } from "./movie-info-modal";
import { MovieVideoModal } from "./movie-video-modal";

interface MovieCardProps {
  movie: SafeMovie;
  focusedComment?: SafeComment | null;
  backButton?: boolean;
}

const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  focusedComment = null,
  backButton = false,
}) => {
  const { data: session } = useSession();

  const router = useRouter();

  const [isReviewModalOpen, setReviewModalOpen] = useState(false);
  const [isVideoModalOpen, setVideoModalOpen] = useState(false);
  const [isCommentDrawerOpen, setCommentDrawerOpen] = useState(false);
  const [isInfoModalOpen, setIndoModalOpen] = useState(false);

  const { setLoginModalOpen } = useAuthModalStore();
  const clearReply = useReplyStore((state) => state.clearReply);

  const { isReviewed, isPending: isReviewing } = useReviewActions(
    movie.movieId
  );

  const {
    isBookmarked,
    isPending: isBookmarking,
    toggleBookmark,
  } = useBookmarkActions(movie.movieId);

  const handleBookmarkClick = () => {
    if (!session) {
      setLoginModalOpen(true);
      return;
    }
    toggleBookmark();
  };

  const handleCommentDrawerChange = (open: boolean) => {
    if (!open) {
      clearReply();
    }
    setCommentDrawerOpen(open);
  };

  useEffect(() => {
    if (focusedComment) {
      setCommentDrawerOpen(true);
    }
  }, [focusedComment]);

  const handleBack = () => {
    router.back();
  };

  return (
    <div className="relative h-full w-full">
      <Image
        src={movie.posterPath}
        alt={movie.title}
        fill
        sizes="(min-width: 1280px) 25vw, (min-width: 768px) 33vw, 50vw"
        className="rounded-md object-cover"
        priority
      />
      {backButton && (
        <Button
          variant="secondary"
          size="icon"
          className="absolute left-4 top-28 z-10 h-12 w-12 rounded-full bg-background md:top-8"
          onClick={handleBack}
        >
          <ChevronLeft size={32} />
        </Button>
      )}
      <AuthButton
        variant="secondary"
        size="icon"
        onClick={() => setReviewModalOpen(true)}
        icon={
          <Star
            size={32}
            className={isReviewed ? "fill-foreground" : "fill-none"}
          />
        }
        className="absolute right-4 top-28 h-12 w-12 rounded-full md:top-8"
        disabled={isReviewing}
      />
      <MovieReviewModal
        movieId={movie.movieId}
        open={isReviewModalOpen}
        onOpenChange={setReviewModalOpen}
      />
      <AuthButton
        variant="secondary"
        size="icon"
        onClick={handleBookmarkClick}
        icon={
          <Bookmark
            size={32}
            className={isBookmarked ? "fill-foreground" : "fill-none"}
          />
        }
        className="absolute right-4 top-48 h-12 w-12 rounded-full md:top-28"
        disabled={isBookmarking}
      />
      <Button
        variant="secondary"
        size="icon"
        className="absolute bottom-52 right-4 h-12 w-12 rounded-full bg-background"
        onClick={() => setVideoModalOpen(true)}
        disabled={!movie.videos || movie.videos.length === 0}
      >
        <Play size={32} />
      </Button>
      <MovieVideoModal
        videos={movie.videos}
        open={isVideoModalOpen}
        onOpenChange={setVideoModalOpen}
      />
      <Button
        variant="secondary"
        size="icon"
        className="absolute bottom-32 right-4 h-12 w-12 rounded-full bg-background"
        onClick={() => setCommentDrawerOpen(true)}
      >
        <MessageSquare size={32} />
      </Button>
      <CommentDrawer
        movieId={movie.movieId}
        focusedComment={focusedComment}
        open={isCommentDrawerOpen}
        onOpenChange={handleCommentDrawerChange}
      />
      <div
        className="absolute bottom-4 left-4 right-4 cursor-pointer rounded bg-background p-2"
        onClick={() => setIndoModalOpen(true)}
      >
        <p className="line-clamp-3 text-sm">{movie.overview}</p>
      </div>
      <MovieInfoModal
        title={movie.title}
        tagline={movie.tagline}
        releaseDate={movie.releaseDate}
        runtime={movie.runtime}
        overview={movie.overview}
        open={isInfoModalOpen}
        onOpenChange={setIndoModalOpen}
      />
    </div>
  );
};

export default MovieCard;
