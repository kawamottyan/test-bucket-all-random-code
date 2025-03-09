import { updateConsent } from "@/actions/update-consent";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useTrainingConsentStore } from "@/stores/use-training-consent-store";
import { useState } from "react";
import { toast } from "sonner";

interface TrainingConsentModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function TrainingConsentModal({
  open,
  onOpenChange,
}: TrainingConsentModalProps) {
  const [isAccepting, setIsAccepting] = useState(false);
  const [isDeclining, setIsDeclining] = useState(false);
  const { setEnabled } = useTrainingConsentStore();

  const handleAccept = async () => {
    setIsAccepting(true);
    try {
      const uuid = localStorage.getItem("uuid");

      if (!uuid) {
        toast.error("UUID not found. Please refresh the page.");
        return;
      }

      await setEnabled(true);
      toast.success("Thank you for helping us improve MovieAI😊");
      onOpenChange(false);

      updateConsent(uuid, true);

    } finally {
      setIsAccepting(false);
    }
  };

  const handleDecline = async () => {
    setIsDeclining(true);
    try {
      const uuid = localStorage.getItem("uuid");

      if (!uuid) {
        toast.error("UUID not found. Please refresh the page.");
        return;
      }

      await setEnabled(false);
      toast.message("You've opted out of AI training");
      onOpenChange(false);

      updateConsent(uuid, false);

    } finally {
      setIsDeclining(false);
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent
        className="max-w-md"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <AlertDialogHeader>
          <AlertDialogTitle>Help Improve Our Services</AlertDialogTitle>
          <AlertDialogDescription className="pt-2">
            We&apos;d like to collect anonymous usage data to understand how our
            services are being used and make improvements.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={handleDecline}
            disabled={isDeclining || isAccepting}
            className="w-full sm:w-auto"
          >
            {isDeclining ? "Declining..." : "Decline"}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={handleAccept}
            disabled={isDeclining || isAccepting}
            className="w-full sm:w-auto"
          >
            {isAccepting ? "Accepting..." : "Enable"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
