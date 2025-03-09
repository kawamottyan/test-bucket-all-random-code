"use client";

import { useCommandBarStore } from "@/stores/commandbar-store";
import { useEffect } from "react";

export function useCommandbar() {
  const { open, setOpen, searchInput, setSearchInput } = useCommandBarStore();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "/") {
        if (
          (e.target instanceof HTMLElement && e.target.isContentEditable) ||
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement ||
          e.target instanceof HTMLSelectElement
        ) {
          return;
        }
        e.preventDefault();
        setOpen(!open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, [open, setOpen]);

  return { open, setOpen, searchInput, setSearchInput };
}
