import { ServerResponse } from "@/types";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { useDebounce } from "use-debounce";
import { DEBOUNCE_TIME } from "../constants";
import { SafeSearchResult } from "../types";

interface SearchResponse extends ServerResponse {
  data?: SafeSearchResult[];
}

export function useSearchResults(searchInput: string) {
  const [results, setResults] = useState<Array<SafeSearchResult>>([]);
  const [debouncedSearchInput] = useDebounce(searchInput, DEBOUNCE_TIME);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        if (debouncedSearchInput.trim()) {
          const params = new URLSearchParams();
          params.append('q', debouncedSearchInput);

          const response = await fetch(`/api/search?${params.toString()}`);

          if (!response.ok) {
            throw new Error(`Failed to fetch search results: ${response.status}`);
          }

          const result: SearchResponse = await response.json();

          if (!result.success) {
            throw new Error(result.message);
          }

          setResults(result.data || []);
        } else {
          setResults([]);
        }
      } catch (error) {
        console.error("Search error:", error);
        toast.error("Sorry, an unexpected error occurred");
      }
    };

    fetchResults();
  }, [debouncedSearchInput]);

  return { results };
}