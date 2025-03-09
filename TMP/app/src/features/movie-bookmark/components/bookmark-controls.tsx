import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search } from "lucide-react";
import { BookmarkSortOption } from "../types";

interface BookmarkControlsProps {
  search: string;
  onSearchChange: (value: string) => void;
  sortBy: BookmarkSortOption;
  onSortChange: (value: BookmarkSortOption) => void;
}

export const BookmarkControls = ({
  search,
  onSearchChange,
  sortBy,
  onSortChange,
}: BookmarkControlsProps) => {
  return (
    <div className="flex items-center justify-between gap-x-4 py-4">
      <div className="relative max-w-sm flex-1">
        <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-8"
        />
      </div>
      <Select value={sortBy} onValueChange={onSortChange}>
        <SelectTrigger className="w-40">
          <SelectValue placeholder="Sort by" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="date-desc">Newest First</SelectItem>
          <SelectItem value="date-asc">Oldest First</SelectItem>
          <SelectItem value="title-asc">Title (A-Z)</SelectItem>
          <SelectItem value="title-desc">Title (Z-A)</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};
