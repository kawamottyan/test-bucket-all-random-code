"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface BookmarkLoadingProps {
  length?: number;
}

export const BookmarkLoading = ({ length = 10 }: BookmarkLoadingProps) => (
  <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
    {Array.from({ length }).map((_, index) => (
      <BookmarkSkeleton key={index} />
    ))}
  </div>
);

export const BookmarkSkeleton = () => (
  <Card className="h-48 p-0">
    <CardContent className="h-full p-0">
      <div className="flex h-full">
        <Skeleton className="h-full w-32 rounded-l-md" />
        <div className="flex-1 p-4">
          <div className="flex flex-col gap-y-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-16 w-full" />
          </div>
        </div>
      </div>
    </CardContent>
  </Card>
);
