"use client";

import { Button } from "@/components/ui/button";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { Bell } from "lucide-react";
import { useState } from "react";
import NotificationIndicator from "./notification-indicator";
import { NotificationModal } from "./notification-modal";

const NotificationSection = () => {
  const { data: user } = useFetchCurrentUser();
  const [isNotificationModalOpen, setNotificationModalOpen] = useState(false);

  return (
    <>
      <Button
        variant="secondary"
        size="icon"
        className="relative rounded-full bg-transparent hover:bg-muted"
        onClick={() => setNotificationModalOpen(true)}
      >
        {user?.unreadNotificationCount ? (
          <>
            <Bell className="h-5 w-5" />
            <NotificationIndicator />
          </>
        ) : (
          <Bell className="h-5 w-5" />
        )}
      </Button>
      <NotificationModal
        open={isNotificationModalOpen}
        onOpenChange={setNotificationModalOpen}
      />
    </>
  );
};

export default NotificationSection;
