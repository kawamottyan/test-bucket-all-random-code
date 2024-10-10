"use client";

import { useState } from "react";
import axios from "axios";
import { SafeComment } from "@/types";

interface UseSubmitCommentProps {
    movieId: number;
}

export const useSubmitComment = ({ movieId }: UseSubmitCommentProps) => {
    const [isSubmitting, setIsSubmitting] = useState(false);

    const submitComment = async (content: string, parentId?: string): Promise<SafeComment> => {
        if (!content.trim()) {
            throw new Error("Content is empty");
        }

        setIsSubmitting(true);

        try {
            const response = await axios.post(`/api/comment/${movieId}`, {
                content,
                parentId,
            });

            return response.data.comment;
        } catch (error) {
            console.error("Error submitting comment:", error);
            throw error;
        } finally {
            setIsSubmitting(false);
        }
    };

    return { isSubmitting, submitComment };
};
