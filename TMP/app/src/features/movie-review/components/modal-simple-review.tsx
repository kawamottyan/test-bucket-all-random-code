import { Star } from "lucide-react";
import { useState } from "react";
import { useReviewActions } from "../hooks/use-review-actions";

interface ModalSimpleReviewProps {
  movieId: number;
  onOpenChange: (open: boolean) => void;
}

export function ModalSimpleReview({
  movieId,
  onOpenChange,
}: ModalSimpleReviewProps) {
  const [hoverRating, setHoverRating] = useState(0);
  const { review, isReviewed, isPending, addReview } =
    useReviewActions(movieId);
  const currentRating = isReviewed ? (review?.rating ?? 0) : 0;

  const onRate = (rating: number) => {
    const today = new Date().toISOString().split("T")[0];

    addReview({
      rating,
      watchDate: today,
    });
    onOpenChange(false);
  };

  return (
    <div className="my-4 flex justify-center">
      <div className="my-8 flex">
        {[1, 2, 3, 4, 5].map((starValue) => (
          <button
            key={starValue}
            disabled={isPending}
            onClick={() => onRate(starValue)}
            onMouseEnter={() => setHoverRating(starValue)}
            onMouseLeave={() => setHoverRating(0)}
          >
            <Star
              size={32}
              className={`mx-2 transition-colors ${
                isPending ? "cursor-not-allowed" : "cursor-pointer"
              } ${
                (
                  hoverRating
                    ? starValue <= hoverRating
                    : starValue <= currentRating
                )
                  ? "fill-foreground"
                  : "fill-none"
              }`}
            />
          </button>
        ))}
      </div>
    </div>
  );
}
