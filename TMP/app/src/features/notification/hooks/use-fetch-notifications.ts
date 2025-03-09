import { NOTIFICATION_QUERY_KEY } from "@/constants/query-keys";
import { ServerResponse } from "@/types";
import { useQuery } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { SafeNotification } from "../types";

interface NotificationsResponse extends ServerResponse {
  data?: SafeNotification[];
}

export const useFetchNotifications = () => {
  const { data: session } = useSession();

  return useQuery<SafeNotification[]>({
    queryKey: NOTIFICATION_QUERY_KEY,
    queryFn: async () => {
      const response = await fetch('/api/user/notifications');

      if (!response.ok) {
        throw new Error(`Failed to fetch notifications: ${response.status}`);
      }

      const result: NotificationsResponse = await response.json();

      if (!result.success) {
        throw new Error(result.message);
      }

      return result.data || [];
    },
    enabled: !!session,
  });
};