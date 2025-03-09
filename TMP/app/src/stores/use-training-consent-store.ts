"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface TrainingConsent {
  isEnabled: boolean;
  setEnabled: (enabled: boolean) => void;
  lastUpdated: string | null;
}

export const useTrainingConsentStore = create<TrainingConsent>()(
  persist(
    (set) => ({
      isEnabled: false,
      lastUpdated: null,
      setEnabled: (enabled) =>
        set({
          isEnabled: enabled,
          lastUpdated: new Date().toISOString(),
        }),
    }),
    {
      name: "training-consent",
    }
  )
);
