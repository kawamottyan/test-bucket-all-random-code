import Container from "@/components/container";
import { CommentType } from "@/features/movie-comment/types";
import MovieCard from "@/features/movie/components/movie-card";
import { Metadata } from "next";
import { notFound } from "next/navigation";

interface RouteParams {
  movieId: string;
}

interface MoviePageProps {
  params: Promise<RouteParams>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}

export async function generateMetadata({
  params,
}: MoviePageProps): Promise<Metadata> {
  const movieId = parseInt(((await params)?.movieId as string) || "", 10);
  const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL}/api/movies/${movieId}`)
  if (!response.ok) {
    notFound();
  }

  const result = await response.json();
  const movie = result.data;

  if (!movie) {
    notFound();
  }

  return {
    title: movie.title,
    description: movie.overview,
    openGraph: {
      images: [
        {
          url: movie.posterPath,
        },
      ],
    },
  };
}

const MoviePage = async ({ params, searchParams }: MoviePageProps) => {
  const movieId = parseInt(((await params)?.movieId as string) || "", 10);
  const searchParamsResolved = await searchParams;

  const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL}/api/movies/${movieId}`)
  if (!response.ok) {
    notFound();
  }

  const result = await response.json();
  const movie = result.data;

  if (!movie) {
    notFound();
  }

  let focusedComment = null;
  if (searchParamsResolved?.s && searchParamsResolved?.t) {
    const slug = searchParamsResolved.s as string;
    const type = searchParamsResolved.t as CommentType;

    const commentResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URLx}/api/movies/${movieId}/comments/${slug}?type=${type}`);
    if (commentResponse.ok) {
      const commentResult = await commentResponse.json();
      focusedComment = commentResult.data;
    }
  }

  return (
    <Container>
      <div className="flex h-dvh items-center justify-center">
        <div className="h-dvh w-full max-w-xl md:h-[calc(100vh-10rem)]">
          <MovieCard movie={movie} focusedComment={focusedComment} backButton />
        </div>
      </div>
    </Container>
  );
};

export default MoviePage;
