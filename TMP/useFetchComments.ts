import { useState, useRef, useCallback } from "react";
import axios from "axios";
import { SafeComment } from "@/types";

interface UseFetchCommentsProps {
    movieId: number;
    take: number;
}

export const useFetchComments = ({ movieId, take }: UseFetchCommentsProps) => {
    const [comments, setComments] = useState<SafeComment[]>([]);
    const [hasMore, setHasMore] = useState(true);
    const skipRef = useRef(0);
    const isFetching = useRef(false);

    const fetchComments = useCallback(async () => {
        if (isFetching.current) return;
        isFetching.current = true;

        try {
            const response = await axios.get("/api/comment", {
                params: {
                    movieId,
                    skip: skipRef.current,
                    take,
                },
            });

            const newComments = response.data.comments;

            setComments((prevComments) => [...prevComments, ...newComments]);
            skipRef.current += take;

            if (newComments.length < take) {
                setHasMore(false);
            }
        } catch (error) {
            console.error("Failed to fetch comments:", error);
        } finally {
            isFetching.current = false;
        }
    }, [movieId, take]);

    return { comments, fetchComments, hasMore };
};
