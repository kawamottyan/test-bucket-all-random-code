import {
  CURRENT_USER_QUERY_KEY,
  NOTIFICATION_QUERY_KEY,
} from "@/constants/query-keys";
import { SafeCurrentUser } from "@/types";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { SafeNotification } from "../types";

export const useNotificationActions = () => {
  const queryClient = useQueryClient();

  const markingAsRead = useMutation({
    mutationFn: async (notificationId: string) => {
      const response = await fetch(
        `/api/notifications/${notificationId}/mark-as-read`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to mark notification as read");
      }
    },
    onSuccess: (_, notificationId) => {
      queryClient.setQueryData<SafeNotification[]>(
        NOTIFICATION_QUERY_KEY,
        (oldNotifications) => {
          if (!oldNotifications) return [];

          return oldNotifications.map((notification) =>
            notification.id === notificationId
              ? { ...notification, readAt: new Date().toISOString() }
              : notification
          );
        }
      );

      queryClient.setQueryData(
        CURRENT_USER_QUERY_KEY,
        (oldData: SafeCurrentUser) => {
          if (!oldData) return oldData;
          return {
            ...oldData,
            unreadNotificationCount: oldData.unreadNotificationCount - 1,
          };
        }
      );
    },
    onError: () => {
      toast.error("Failed to mark notification as read");
    },
  });

  const markingAllAsAchieved = useMutation({
    mutationFn: async () => {
      const response = await fetch("/api/notifications/mark-all-as-achieved", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to mark notifications as achieved");
      }
    },
    onSuccess: () => {
      queryClient.setQueryData<SafeNotification[]>(NOTIFICATION_QUERY_KEY, []);

      queryClient.setQueryData(
        CURRENT_USER_QUERY_KEY,
        (oldData: SafeCurrentUser) => {
          if (!oldData) return oldData;
          return {
            ...oldData,
            unreadNotificationCount: 0,
          };
        }
      );

      toast.success("All notifications marked as achieved");
    },
    onError: () => {
      toast.error("Failed to mark notifications as achieved");
    },
  });

  return {
    markAsRead: markingAsRead.mutate,
    markAllAsAchieved: markingAllAsAchieved.mutate,
    isPending: markingAsRead.isPending || markingAllAsAchieved.isPending,
  };
};
