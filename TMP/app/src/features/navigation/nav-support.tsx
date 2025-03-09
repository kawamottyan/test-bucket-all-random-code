"use cllient";

import { AlertDialog } from "@/components/ui/alert-dialog";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { FeedbackModal } from "@/features/support/components/feedback-modal";
import { TrainingConsentModal } from "@/features/support/components/training-consent-modal";
import { Bot, Send } from "lucide-react";
import { useState } from "react";

export function NavSupport() {
  const [isConsentModalOpen, setConsentModalOpen] = useState(false);
  const [isFeedbackModalOpen, setFeedbackModalOpen] = useState(false);
  return (
    <SidebarMenu className="gap-y-1">
      <SidebarMenuItem>
        <AlertDialog>
          <SidebarMenuButton asChild onClick={() => setConsentModalOpen(true)}>
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              <span>Training Consent</span>
            </div>
          </SidebarMenuButton>
          <TrainingConsentModal
            open={isConsentModalOpen}
            onOpenChange={setConsentModalOpen}
          />
        </AlertDialog>
      </SidebarMenuItem>
      <SidebarMenuItem>
        <SidebarMenuButton asChild onClick={() => setFeedbackModalOpen(true)}>
          <div className="flex items-center gap-2">
            <Send className="h-4 w-4" />
            <span>Feedback</span>
          </div>
        </SidebarMenuButton>
        <FeedbackModal
          open={isFeedbackModalOpen}
          onOpenChange={setFeedbackModalOpen}
        />
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
