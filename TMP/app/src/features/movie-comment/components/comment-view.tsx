import { useEffect } from "react";
import { useInView } from "react-intersection-observer";
import { useFetchComments } from "../hooks/use-fetch-comments";
import CommentDisplayCard from "./comment-display-card";
import CommentSkeleton from "./comment-skeleton";

interface CommentsViewProps {
  movieId: number;
  focusedId: string | null;
  open: boolean;
}

const CommentsView: React.FC<CommentsViewProps> = ({
  movieId,
  focusedId,
  open,
}) => {
  const { ref, inView } = useInView();

  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isPending } =
    useFetchComments({ movieId, parentId: null, enabled: open });

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const comments = data?.pages.flatMap((page) => page.comments) ?? [];

  if (isPending) {
    return (
      <div className="flex flex-col gap-y-4">
        <CommentSkeleton />
      </div>
    );
  }

  if (comments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center px-4 py-12 text-center">
        <div className="mb-2 text-lg font-semibold">No comments yet</div>
        <p className="max-w-sm">
          Be the first to share your thoughts about this movie. Start the
          conversation!
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col pr-4">
      {comments.map((comment) => (
        <CommentDisplayCard
          key={comment.id}
          movieId={movieId}
          comment={comment}
          parentId={null}
          focusedId={focusedId}
          depth={1}
        />
      ))}
      {hasNextPage && (
        <div ref={ref} className="flex h-8 w-full items-center justify-center">
          {isFetchingNextPage && <CommentSkeleton />}
        </div>
      )}
    </div>
  );
};

export default CommentsView;
