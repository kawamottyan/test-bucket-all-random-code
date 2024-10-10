import { useRef, useCallback } from "react";

export const useInfiniteScroll = (hasMore: boolean, fetchMore: () => void) => {
    const observer = useRef<IntersectionObserver | null>(null);

    const lastElementRef = useCallback(
        (node: HTMLDivElement | null) => {
            if (observer.current) observer.current.disconnect();
            observer.current = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting && hasMore) {
                    fetchMore();
                }
            });
            if (node) observer.current.observe(node);
        },
        [hasMore, fetchMore]
    );

    return { lastElementRef };
};