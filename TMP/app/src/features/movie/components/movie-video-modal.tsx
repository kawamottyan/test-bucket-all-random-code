"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Carousel,
  CarouselApi,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { SafeVideo } from "../types";

interface MovieVideoModalProps {
  videos: SafeVideo[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MovieVideoModal({
  videos,
  open,
  onOpenChange,
}: MovieVideoModalProps) {
  const [carouselApi, setCarouselApi] = useState<CarouselApi | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [canScrollPrev, setCanScrollPrev] = useState(false);
  const [canScrollNext, setCanScrollNext] = useState(false);
  const currentVideo = videos[currentIndex];

  useEffect(() => {
    if (carouselApi) {
      const onSelect = () => {
        setCurrentIndex(carouselApi.selectedScrollSnap());
        setCanScrollPrev(carouselApi.canScrollPrev());
        setCanScrollNext(carouselApi.canScrollNext());
      };
      onSelect();
      carouselApi.on("select", onSelect);

      return () => {
        carouselApi.off("select", onSelect);
      };
    }
  }, [carouselApi]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{currentVideo?.name}</DialogTitle>
        </DialogHeader>
        <div className="flex gap-2">
          {currentVideo?.official && (
            <Badge variant="outline" className="w-fit">
              Official
            </Badge>
          )}
          {currentVideo?.type && (
            <Badge variant="outline" className="w-fit">
              {currentVideo.type}
            </Badge>
          )}
        </div>
        <div className="w-full overflow-hidden">
          <Carousel setApi={setCarouselApi} className="w-full">
            <CarouselContent>
              {videos.map((video, index) => (
                <CarouselItem key={`${video.id}-${index}`}>
                  {index === currentIndex ? (
                    <div className="relative aspect-video w-full">
                      <iframe
                        className="absolute inset-0 h-full w-full"
                        src={video.videoPath}
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                        allowFullScreen
                      />
                    </div>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                </CarouselItem>
              ))}
            </CarouselContent>
          </Carousel>

          <div className="my-4 flex justify-between">
            <Button
              className="h-8 w-8 rounded-full"
              variant="outline"
              onClick={() => carouselApi?.scrollPrev()}
              disabled={!canScrollPrev}
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="sr-only">Previous slide</span>
            </Button>

            <Button
              className="h-8 w-8 rounded-full"
              variant="outline"
              onClick={() => carouselApi?.scrollNext()}
              disabled={!canScrollNext}
            >
              <ArrowRight className="h-4 w-4" />
              <span className="sr-only">Next slide</span>
            </Button>
          </div>
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">Close</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
