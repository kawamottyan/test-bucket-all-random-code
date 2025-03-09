import { Notification } from "@prisma/client";

export type SafeNotification = Omit<
  Notification,
  | "readAt"
  | "achievedAt"
  | "deletedAt"
  | "createdAt"
  | "updatedAt"
  | "userId"
  | "senderId"
> & {
  createdAt: string;
  readAt: string | null;
  achievedAt: string | null;
  deletedAt: string | null;
};
