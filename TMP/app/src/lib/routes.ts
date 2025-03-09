export const publicRoutes = [
  "/",
  "/auth/(.*)",
  "/movies/(.*)",
  "/bookmark",
  "/review",
  "/rate-limit"
];

export const authRoutes = ["/setting", "/setting/(.*)"];

export const apiAuthPrefix = "/api/auth";

export const LOGIN_PAGE = "/auth/login";
