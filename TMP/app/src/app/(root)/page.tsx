"use client";

import Container from "@/components/container";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import { MOVIE_ID_QUERY_KEY } from "@/constants/query-keys";
import MovieCard from "@/features/movie/components/movie-card";
import { useCarousel } from "@/features/movie/hooks/use-carousel";
import { useFetchInfiniteMovies } from "@/features/movie/hooks/use-fetch-infinite-movies";
import { useVisibilityObserver } from "@/features/movie/hooks/use-visibility-observer";
import Loading from "../loading";

export default function Home() {
  const { movies, fetchNextPage, isFetchingNextPage } =
    useFetchInfiniteMovies();
  const { setApi } = useCarousel(movies, fetchNextPage, isFetchingNextPage);
  useVisibilityObserver(movies);

  if (!movies.length) {
    return <Loading />;
  }

  return (
    <Container>
      <div className="flex h-dvh w-full items-center justify-center">
        <Carousel
          setApi={setApi}
          opts={{
            align: "start",
          }}
          orientation="vertical"
          className="w-full max-w-full overflow-hidden md:max-w-xl"
        >
          <CarouselContent className="h-dvh md:h-[calc(100vh-10rem)]">
            {movies.map((movie) => (
              <CarouselItem
                key={movie.movieId}
                {...{ [MOVIE_ID_QUERY_KEY]: movie.movieId }}
              >
                <MovieCard movie={movie} />
              </CarouselItem>
            ))}
          </CarouselContent>
        </Carousel>
      </div>
    </Container>
  );
}
