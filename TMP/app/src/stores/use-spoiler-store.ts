import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SpoilerState {
  spoilerSettings: Record<number, boolean>;
  showSpoiler: (movieId: number) => boolean;
  toggleSpoiler: (movieId: number) => void;
}

export const useSpoilerStore = create<SpoilerState>()(
  persist(
    (set, get) => ({
      spoilerSettings: {},
      showSpoiler: (movieId) => get().spoilerSettings[movieId] ?? false,
      toggleSpoiler: (movieId) =>
        set((state) => ({
          spoilerSettings: {
            ...state.spoilerSettings,
            [movieId]: !state.spoilerSettings[movieId],
          },
        })),
    }),
    {
      name: "spoiler-settings",
    }
  )
);
