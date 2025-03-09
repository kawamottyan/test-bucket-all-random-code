import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { formatTimeToNow } from "@/lib/formatter";
import React from "react";
import { SafeComment } from "../types";

interface CommentUserSectionProps {
  user: SafeComment["user"];
  depth: number;
  createdAt: string;
}

const CommentUserSection: React.FC<CommentUserSectionProps> = ({
  user,
  depth,
  createdAt,
}) => {
  const relativeTime = formatTimeToNow(new Date(createdAt));

  return (
    <div className="mb-2 flex items-start space-x-2">
      <Avatar className={depth === 1 ? "h-6 w-6" : "mt-1 h-4 w-4"}>
        {user.image ? (
          <AvatarImage src={user.image} alt={user.username} />
        ) : (
          <AvatarFallback>
            {(user.name ?? "").slice(0, 1).toUpperCase() || "U"}
          </AvatarFallback>
        )}
      </Avatar>
      <div className="flex items-center space-x-2">
        <p className="text-sm font-medium">{user.username}</p>
        <p className="text-sm font-light text-muted-foreground">
          {relativeTime}
        </p>
      </div>
    </div>
  );
};

export default CommentUserSection;
