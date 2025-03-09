import { User } from "@prisma/client";

export type ServerResponse = {
  success: boolean;
  message: string;
};

export type SafeCurrentUser = Omit<
  User,
  "password" | "emailVerified" | "createdAt" | "updatedAt"
> & {
  hasPassword: boolean;
  emailVerified: string | null;
  createdAt: string;
  updatedAt: string;
  accounts: SafeAccount[];
};

type SafeAccount = {
  provider: string;
};

export type InteractionType = 'poster_viewed' | 'detail_viewed' | 'play_started';

export type SafeInteractionLog = {
  uuid?: string;
  interaction_type: InteractionType;
  item_id?: string;
  watch_time?: string;
  query?: string;
  created_at: string;
  local_timestamp: string,
  index?: string,
}

export interface SessionInfo {
  session_id: string
  timezone: string;
  device: string;
  ip: string;
  browser: string;
  os: string;
  createdAt: string;
}