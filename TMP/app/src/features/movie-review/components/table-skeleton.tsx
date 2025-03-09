import { Skeleton } from "@/components/ui/skeleton";
import { TableCell, TableRow } from "@/components/ui/table";

interface TableSkeletonProps {
  rowCount: number;
  columnCount: number;
}

export const TableSkeleton = ({
  rowCount,
  columnCount,
}: TableSkeletonProps) => {
  return (
    <>
      {[...Array(rowCount)].map((_, index) => (
        <TableRow key={index}>
          {[...Array(columnCount)].map((_, cellIndex) => (
            <TableCell key={cellIndex}>
              <Skeleton className="h-6 w-full" />
            </TableCell>
          ))}
        </TableRow>
      ))}
    </>
  );
};
