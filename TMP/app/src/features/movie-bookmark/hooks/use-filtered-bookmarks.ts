import { useState } from "react";
import { BookmarkSortOption } from "../types";
import { useFetchBookmarks } from "./use-fetch-bookmarks";

export const useFilteredBookmarks = () => {
  "use memo";

  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<BookmarkSortOption>("date-desc");
  const { data: bookmarks, isPending } = useFetchBookmarks();

  const filtered =
    bookmarks?.filter((bookmark) =>
      bookmark.movie.title.toLowerCase().includes(search.toLowerCase())
    ) ?? [];

  const sortedBookmarks = filtered.sort((a, b) => {
    switch (sortBy) {
      case "date-desc":
        return (
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );
      case "date-asc":
        return (
          new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
        );
      case "title-asc":
        return a.movie.title.localeCompare(b.movie.title);
      case "title-desc":
        return b.movie.title.localeCompare(a.movie.title);
      default:
        return 0;
    }
  });

  return {
    bookmarks: sortedBookmarks,
    isPending,
    search,
    setSearch,
    sortBy,
    setSortBy,
  };
};
