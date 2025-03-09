import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  MAX_NESTABLE_DEPTH,
  MAX_TRUNCATE_COMMENT_LENGTH,
} from "@/features/movie-comment/constants";
import { truncateText } from "@/lib/formatter";
import { useReplyStore } from "@/stores/use-reply-store";
import { X } from "lucide-react";
import React from "react";

const ReplyContextCard: React.FC = () => {
  const { replyContext, clearReply } = useReplyStore();

  if (!replyContext) {
    return null;
  }

  const isMaxDepth = replyContext.depth >= MAX_NESTABLE_DEPTH;
  const replyText = isMaxDepth ? "Mentioning" : "Replying to";

  return (
    <Card className="p-4">
      <div className="flex flex-col">
        <div className="flex items-start justify-between">
          <p className="text-sm font-medium text-muted-foreground">
            {replyText} @{replyContext.username}
          </p>
          <Button
            variant="ghost"
            size="icon"
            className="-mr-2 -mt-2"
            onClick={clearReply}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          {truncateText(replyContext.content, MAX_TRUNCATE_COMMENT_LENGTH)}
        </p>
      </div>
    </Card>
  );
};

export default ReplyContextCard;
