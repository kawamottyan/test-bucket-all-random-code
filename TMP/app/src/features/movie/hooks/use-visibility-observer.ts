import { MOVIE_ID_QUERY_KEY } from "@/constants/query-keys";
import { useEffect, useRef, useState } from "react";
import { logInteraction } from "../actions/log-interaction";
import { VISIBILITY_THRESHOLD } from "../constants";
import { SafeMovie } from "../types";


export const useVisibilityObserver = (
  movies: SafeMovie[],
  threshold = VISIBILITY_THRESHOLD
) => {
  const timeStamps = useRef<Record<number, number>>({});
  const observerRef = useRef<IntersectionObserver | null>(null);
  const containerRef = useRef<HTMLElement | null>(null);
  const [uuid, setUuid] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const storedUuid = localStorage.getItem('uuid');
      setUuid(storedUuid);
    }
  }, []);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const movieId = entry.target.getAttribute(MOVIE_ID_QUERY_KEY);
          const movieIdNumber = movieId ? Number(movieId) : null;
          if (movieIdNumber == null) return;

          const index = movies.findIndex(movie => movie.movieId === movieIdNumber);

          if (entry.isIntersecting) {
            timeStamps.current[movieIdNumber] = Date.now();
          } else {
            const startTime = timeStamps.current[movieIdNumber];
            if (startTime) {
              const watchTime = (Date.now() - startTime) / 1000;
              delete timeStamps.current[movieIdNumber];

              logInteraction({
                interactionType: "poster_viewed",
                timestamp: String(Date.now()),
                itemId: `movie:${movieIdNumber}`,
                watchTime: String(watchTime),
                uuid: uuid || undefined,
                index: index !== -1 ? String(index) : undefined
              })

            }
          }
        });
      },
      { threshold, root: containerRef.current || null }
    );

    const elements = containerRef.current
      ? containerRef.current.querySelectorAll(`[${MOVIE_ID_QUERY_KEY}]`)
      : document.querySelectorAll(`[${MOVIE_ID_QUERY_KEY}]`);

    elements.forEach((element) => observerRef.current?.observe(element));

    return () => {
      observerRef.current?.disconnect();
    };
  }, [movies, threshold, uuid]);

  return { containerRef };
};