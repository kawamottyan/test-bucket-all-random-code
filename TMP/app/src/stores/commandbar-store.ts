import { create } from "zustand";

interface CommandBarState {
  open: boolean;
  searchInput: string;
  setOpen: (open: boolean) => void;
  setSearchInput: (input: string) => void;
}

export const useCommandBarStore = create<CommandBarState>((set) => ({
  open: false,
  searchInput: "",
  setOpen: (open) => set({ open }),
  setSearchInput: (input) => set({ searchInput: input }),
}));
