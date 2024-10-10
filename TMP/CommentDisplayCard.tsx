import React, { useState } from "react";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Card } from "../ui/card";
import { Button } from "@/components/ui/button";
import { SafeComment, SafeUser } from "@/types";
import { formatTimeToNow } from "../utils/dateFormatter";
import { CircleUser, Heart, MessageCircle, Ellipsis } from 'lucide-react';
import CommentPost from "./CommentPost";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../ui/collapsible";

interface CommentDisplayCardProps {
    comment: SafeComment;
    currentUser: SafeUser | null;
    movieId: number;
    onCommentSubmit: (newComment: SafeComment) => void;
}

const CommentDisplayCard: React.FC<CommentDisplayCardProps> = ({ comment, currentUser, movieId, onCommentSubmit }) => {
    const [isReplying, setIsReplying] = useState(false);
    const [isRepliesOpen, setIsRepliesOpen] = useState(false);
    const relativeTime = formatTimeToNow(new Date(comment.createdAt));

    console.log('comment', comment)

    return (
        <Card className="px-4 py-2 border-none">
            <div className="flex items-start space-x-2">
                <Avatar className="h-6 w-6">
                    {comment.user.image ? (
                        <AvatarImage src={comment.user.image} alt={`${comment.user.username}'s avatar`} />
                    ) : (
                        <AvatarFallback>
                            <CircleUser className="h-6 w-6" />
                        </AvatarFallback>
                    )}
                </Avatar>
                <div className="flex-1">
                    <div className="flex items-center space-x-2">
                        <p className="text-sm font-medium">{comment.user.username}</p>
                        <p className="text-sm text-muted-foreground font-light">{relativeTime}</p>
                    </div>
                    <p className="text-sm">{comment.content}</p>
                </div>
                <div className="ml-auto flex">
                    <Button variant="ghost" size="icon">
                        <Heart className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon" onClick={() => setIsReplying(!isReplying)}>
                        <MessageCircle className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="icon">
                        <Ellipsis className="h-4 w-4" />
                    </Button>
                </div>
            </div>

            {isReplying && (
                <div className="mt-4">
                    <CommentPost
                        currentUser={currentUser}
                        movieId={movieId}
                        onCommentSubmit={onCommentSubmit}
                        parentId={comment.id}
                    />
                </div>
            )}
            {comment._count?.replies > 0 && (
                <Collapsible open={isRepliesOpen} onOpenChange={setIsRepliesOpen} className="mt-2">
                    <CollapsibleTrigger asChild>
                        <Button variant="ghost" size="sm">
                            {isRepliesOpen
                                ? `Hide replies`
                                : `Show ${comment._count.replies} more replies`}
                        </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                        {/* リプライのリストを表示 */}
                        {/* {comment.replies?.map((reply) => (
                            <CommentDisplayCard
                                key={reply.id}
                                comment={reply}
                                currentUser={currentUser}
                                movieId={movieId}
                                onCommentSubmit={onCommentSubmit}
                            />
                        ))} */}
                    </CollapsibleContent>
                </Collapsible>
            )}
        </Card>
    );
};

export default CommentDisplayCard;
