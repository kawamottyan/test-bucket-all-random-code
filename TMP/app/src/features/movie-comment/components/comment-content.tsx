"use client";

import { useSpoilerStore } from "@/stores/use-spoiler-store";
import { MENTION_REGEX } from "../constants";

interface CommentContentProps {
  movieId: number;
  content: string;
  isSpoiler: boolean;
  isFocused: boolean;
  isDeleted: boolean;
}

const CommentContent = ({
  movieId,
  content,
  isSpoiler,
  isFocused,
  isDeleted,
}: CommentContentProps) => {
  const showSpoiler = useSpoilerStore((state) => state.showSpoiler(movieId));

  const baseStyles = `text-sm overflow-wrap-break-word ${isFocused ? "font-bold" : ""}`;
  const mutedStyles = "text-sm overflow-wrap-break-word text-muted-foreground";

  if (isDeleted) {
    return <p className={mutedStyles}>[deleted]</p>;
  }

  if (isSpoiler && !showSpoiler) {
    return <p className={mutedStyles}>[spoiler]</p>;
  }

  const renderContent = () => {
    const elements: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    while ((match = MENTION_REGEX.exec(content)) !== null) {
      if (match.index > lastIndex) {
        elements.push(
          <span key={`text-${lastIndex}`}>
            {content.slice(lastIndex, match.index)}
          </span>
        );
      }

      elements.push(
        <span key={`mention-${match.index}`} className="text-muted-foreground">
          {match[0]}
        </span>
      );

      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      elements.push(
        <span key={`text-${lastIndex}`}>{content.slice(lastIndex)}</span>
      );
    }

    return elements;
  };

  return <p className={baseStyles}>{renderContent()}</p>;
};

export default CommentContent;
