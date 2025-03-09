import { formatTimeToNow } from "@/lib/formatter";
import Link from "next/link";
import React from "react";
import { useNotificationActions } from "../hooks/use-notification-actions";
import { SafeNotification } from "../types";

interface NotificationItemProps {
  notification: SafeNotification;
  onOpenChange: (open: boolean) => void;
}

const NotificationItem: React.FC<NotificationItemProps> = ({
  notification,
  onOpenChange,
}) => {
  const { markAsRead } = useNotificationActions();

  const handleClick = () => {
    if (notification.link) {
      if (!notification.readAt) {
        markAsRead(notification.id);
      }
      onOpenChange(false);
    }
  };
  const Content = (
    <div className="mr-4 grid cursor-pointer grid-cols-[25px_1fr] items-center rounded-md p-4 hover:bg-muted">
      <span
        className={`h-2 w-2 rounded-full ${
          notification.link && !notification.readAt ? "bg-primary" : "bg-muted"
        }`}
      />
      <div className="space-y-1">
        <p className="font-small text-sm">{notification.message}</p>
        <p className="text-xs text-muted-foreground">
          {formatTimeToNow(new Date(notification.createdAt), true)}
        </p>
      </div>
    </div>
  );

  if (notification.link) {
    return (
      <Link href={notification.link} onClick={handleClick}>
        {Content}
      </Link>
    );
  }

  return Content;
};

export default NotificationItem;
