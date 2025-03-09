import { SafeNotification } from "../types";
import NotificationItem from "./notification-item";

interface NotificationListProps {
  notifications: SafeNotification[];
  onOpenChange: (open: boolean) => void;
}

const NotificationList = ({
  notifications,
  onOpenChange,
}: NotificationListProps) => {
  if (!notifications?.length) {
    return (
      <div className="flex h-full min-h-24 items-center justify-center">
        <p className="text-center text-sm text-muted-foreground">
          No notifications available.
        </p>
      </div>
    );
  }

  return notifications.map((notification) => (
    <NotificationItem
      key={notification.id}
      notification={notification}
      onOpenChange={onOpenChange}
    />
  ));
};

export default NotificationList;
