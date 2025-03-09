"use client";

import { TrainingConsentModal } from "@/features/support/components/training-consent-modal";
import { useTrainingConsentStore } from "@/stores/use-training-consent-store";
import { useEffect, useState } from "react";

export function TrainingConsentProvider() {
  const [mounted, setMounted] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const { lastUpdated } = useTrainingConsentStore();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (mounted && lastUpdated === null) {
      setShowModal(true);
    }
  }, [lastUpdated, mounted]);

  if (!mounted) {
    return null;
  }

  return (
    <TrainingConsentModal
      open={showModal}
      onOpenChange={(open: boolean) => {
        setShowModal(open);
      }}
    />
  );
}
