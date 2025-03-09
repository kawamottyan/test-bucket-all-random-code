import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ColumnDef } from "@tanstack/react-table";
import { ArrowUpDown, MoreHorizontal, Pencil } from "lucide-react";
import { useState } from "react";
import { SafeReview } from "../types";
import { MovieReviewModal } from "./movie-review-modal";

const ActionsCell = ({ review }: { review: SafeReview }) => {
  const [open, setOpen] = useState(false);

  return (
    <>
      <DropdownMenu modal={false}>
        <DropdownMenuTrigger asChild>
          <MoreHorizontal className="h-4 w-4 cursor-pointer" />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem
            className="cursor-pointer"
            onClick={() => setOpen(true)}
          >
            <Pencil className="mr-2 h-4 w-4" />
            Edit
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <MovieReviewModal
        movieId={review.movie.movieId}
        open={open}
        onOpenChange={setOpen}
      />
    </>
  );
};

export const columns: ColumnDef<SafeReview>[] = [
  {
    id: "title",
    accessorFn: (row) => row.movie.title,
    header: ({ column }) => (
      <div className="flex items-center">
        Title
        <Button
          variant="ghost"
          size="sm"
          className="ml-2 h-8 w-8 p-0"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          <ArrowUpDown className="h-4 w-4" />
        </Button>
      </div>
    ),
  },
  {
    accessorKey: "rating",
    header: ({ column }) => (
      <div className="flex items-center">
        Rating
        <Button
          variant="ghost"
          size="sm"
          className="ml-2 h-8 w-8 p-0"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          <ArrowUpDown className="h-4 w-4" />
        </Button>
      </div>
    ),
  },
  {
    id: "actions",
    cell: ({ row }) => <ActionsCell review={row.original} />,
  },
];
