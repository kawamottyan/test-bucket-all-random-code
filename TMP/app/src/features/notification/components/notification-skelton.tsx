import { Skeleton } from "@/components/ui/skeleton";
import React from "react";

interface NotificationSkeletonProps {
  count?: number;
}

const NotificationSkeleton: React.FC<NotificationSkeletonProps> = ({
  count = 3,
}) => {
  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <div key={index} className="grid grid-cols-[25px_1fr] items-center p-4">
          <Skeleton className="h-2 w-2 rounded-full" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-[80%]" />
            <Skeleton className="h-3 w-[40%]" />
          </div>
        </div>
      ))}
    </>
  );
};

export default NotificationSkeleton;
