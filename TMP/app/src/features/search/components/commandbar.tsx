"use client";

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Film } from "lucide-react";
import { useRouter } from "next/navigation";
import { useCommandbar } from "../hooks/use-commandbar";
import { useSearchResults } from "../hooks/use-search-results";

export function Commandbar() {
  const router = useRouter();
  const { open, setOpen, searchInput, setSearchInput } = useCommandbar();
  const { results } = useSearchResults(searchInput);

  const handleSelect = (movieId: number) => {
    router.push(`/movies/${movieId}`);
    setOpen(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput
        placeholder="Search..."
        value={searchInput}
        onValueChange={setSearchInput}
      />
      <CommandList>
        {searchInput.trim().length === 0 ? (
          <CommandEmpty>Type a search term to begin</CommandEmpty>
        ) : results.length === 0 ? (
          <CommandEmpty>No results found.</CommandEmpty>
        ) : (
          <CommandGroup heading="Movies">
            {results.map((result) => (
              <CommandItem
                key={result.movieId}
                value={result.title}
                className="cursor-pointer"
                onSelect={() => handleSelect(result.movieId)}
              >
                <Film className="mr-2 h-4 w-4" />
                {result.title}
              </CommandItem>
            ))}
          </CommandGroup>
        )}
      </CommandList>
    </CommandDialog>
  );
}
