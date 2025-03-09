"use client";

import { Button } from "@/components/ui/button";
import { useSpoilerStore } from "@/stores/use-spoiler-store";
import { Eye, EyeOff } from "lucide-react";

interface CommentFilterProps {
  movieId: number;
}

const CommentFilter: React.FC<CommentFilterProps> = ({ movieId }) => {
  const showSpoiler = useSpoilerStore((state) => state.showSpoiler(movieId));
  const toggleSpoiler = useSpoilerStore((state) => state.toggleSpoiler);

  const handleToggle = () => {
    toggleSpoiler(movieId);
  };

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={handleToggle}
      className="h-8 w-8"
      title={showSpoiler ? "Hide spoilers" : "Show spoilers"}
    >
      {showSpoiler ? (
        <Eye className="h-4 w-4" />
      ) : (
        <EyeOff className="h-4 w-4" />
      )}
    </Button>
  );
};

export default CommentFilter;
