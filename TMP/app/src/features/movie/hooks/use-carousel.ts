import { CarouselApi } from "@/components/ui/carousel";
import { useCallback, useEffect, useRef, useState } from "react";
import { REMAINING_SLIDES_THRESHOLD, WHEEL_SENSITIVITY } from "../constants";
import { SafeMovie } from "../types";

export const useCarousel = (
  movies: SafeMovie[],
  fetchNextPage: () => void,
  isFetchingNextPage: boolean
) => {
  "use memo";
  const [api, setApi] = useState<CarouselApi | null>(null);
  const isWheelingRef = useRef(false);

  const handleSelect = useCallback(() => {
    if (!api || isFetchingNextPage) {
      return;
    }

    const selectedIndex = api.selectedScrollSnap();
    const remainingSlides = movies.length - selectedIndex;

    if (remainingSlides < REMAINING_SLIDES_THRESHOLD) {
      fetchNextPage();
    }
  }, [api, movies, fetchNextPage, isFetchingNextPage]);

  const handleWheel = useCallback(
    (event: Event) => {
      if (!api || isWheelingRef.current) return;

      const wheelEvent = event as WheelEvent;

      isWheelingRef.current = true;
      setTimeout(() => {
        isWheelingRef.current = false;
      }, 200);

      const delta = wheelEvent.deltaY;
      const currentIndex = api.selectedScrollSnap();

      if (delta < -WHEEL_SENSITIVITY && currentIndex > 0) {
        api.scrollTo(currentIndex - 1);
      } else if (delta > WHEEL_SENSITIVITY && currentIndex < movies.length - 1) {
        api.scrollTo(currentIndex + 1);
      }

      event.preventDefault();
    },
    [api, movies.length]
  );

  useEffect(() => {
    if (!api) return;

    api.on("select", handleSelect);

    const carouselElement = document.querySelector(
      '[role="region"][aria-roledescription="carousel"]'
    );

    if (carouselElement) {
      carouselElement.addEventListener("wheel", handleWheel, {
        passive: false,
      });
    }

    return () => {
      api.off("select", handleSelect);
      if (carouselElement) {
        carouselElement.removeEventListener("wheel", handleWheel);
      }
    };
  }, [api, handleSelect, handleWheel]);

  return { api, setApi };
};
