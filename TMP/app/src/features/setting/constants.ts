export const MIN_USERNAME_LENGTH = 8;
export const MAX_USERNAME_LENGTH = 15;
export const ALLOWED_FILE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
];
export const MAX_FILE_SIZE = 5 * 1024 * 1024;

export const sidebarItems = [
  {
    title: "Account",
    id: "account" as const,
  },
  {
    title: "Profile",
    id: "profile" as const,
  },
  {
    title: "Preference",
    id: "preference" as const,
  },
] as const;

export type TabId = (typeof sidebarItems)[number]["id"];
