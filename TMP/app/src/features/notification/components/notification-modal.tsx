import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Check } from "lucide-react";
import { useFetchNotifications } from "../hooks/use-fetch-notifications";
import { useNotificationActions } from "../hooks/use-notification-actions";
import NotificationList from "./notification-list";
import NotificationSkeleton from "./notification-skelton";
import NotificationToggle from "./notification-toggle";

interface NotificationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function NotificationModal({
  open,
  onOpenChange,
}: NotificationModalProps) {
  const { data: notifications, isPending: isFetchingNotifications } =
    useFetchNotifications();
  const {
    markAsRead,
    markAllAsAchieved,
    isPending: isSubmitting,
  } = useNotificationActions();

  const unreadNotifications =
    notifications?.filter((notification) => notification.readAt === null) ?? [];

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) {
      const unreadNotificationsWithoutLink =
        notifications?.filter(
          (notification) => !notification.readAt && !notification.link
        ) ?? [];

      unreadNotificationsWithoutLink.forEach((notification) => {
        markAsRead(notification.id);
      });
    }
    onOpenChange(newOpen);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Notifications</DialogTitle>
          <DialogDescription>
            You have {unreadNotifications.length} unread messages.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-2">
          <NotificationToggle />
          <ScrollArea className="mt-3 max-h-56">
            {isFetchingNotifications ? (
              <NotificationSkeleton />
            ) : (
              <NotificationList
                notifications={notifications ?? []}
                onOpenChange={onOpenChange}
              />
            )}
          </ScrollArea>
        </div>
        <DialogFooter>
          <Button
            className="w-full"
            onClick={() => markAllAsAchieved()}
            disabled={!notifications?.length || isSubmitting}
          >
            <Check className="mr-2 h-4 w-4" />
            {isSubmitting ? "Submitting..." : "Mark all as achieved"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
