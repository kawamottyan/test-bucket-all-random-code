import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import React from "react";

interface CommentSkeletonProps {
  count?: number;
  isReplies?: boolean;
}

const CommentSkeleton: React.FC<CommentSkeletonProps> = ({
  count = 3,
  isReplies = false,
}) => {
  return (
    <>
      {[...Array(count)].map((_, index) => (
        <Card key={index} className="border-none px-4 pt-4">
          <div className="flex items-start space-x-2">
            <Skeleton
              className={`rounded-full ${isReplies ? "h-4 w-4" : "h-6 w-6"}`}
            />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4" />
              <Skeleton className="h-3" />
            </div>
          </div>
        </Card>
      ))}
    </>
  );
};

export default CommentSkeleton;
