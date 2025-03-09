"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

interface MovieInfoModalProps {
  title: string;
  tagline: string | null;
  releaseDate: string | null;
  runtime: string | null;
  overview: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MovieInfoModal({
  title,
  tagline,
  releaseDate,
  runtime,
  overview,
  open,
  onOpenChange,
}: MovieInfoModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{tagline}</DialogDescription>
        </DialogHeader>
        <ScrollArea>
          <div className="space-y-2">
            <p className="text-sm">{overview}</p>
            <div>
              <p className="text-sm text-muted-foreground">
                Release Date: {releaseDate ? releaseDate : "N/A"}
              </p>
              <p className="text-sm text-muted-foreground">
                Runtime: {runtime !== null ? `${runtime} minutes` : "N/A"}
              </p>
            </div>
          </div>
        </ScrollArea>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
