import { headers } from "next/headers";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { auth } from "./lib/auth";
import ratelimit from "./lib/ratelimit";
import {
  apiAuthPrefix,
  authRoutes,
  LOGIN_PAGE,
  publicRoutes,
} from "./lib/routes";

export default async function middleware(request: NextRequest) {
  const response = NextResponse.next();

  const ip = (await headers()).get("x-forwarded-for") || "127.0.0.1";
  const { success } = await ratelimit.limit(ip);
  if (!success) {
    return NextResponse.redirect(new URL("/rate-limit", request.url));
  }

  if (!request.cookies.has('session_id')) {
    response.cookies.set({
      name: 'session_id',
      value: uuidv4(),
      httpOnly: true,
      sameSite: 'strict',
      maxAge: 60 * 60 * 24 * 7,
    });
  }

  const { nextUrl } = request;
  const isLoggedIn = await auth();

  const pathname = nextUrl.pathname;

  const isApiAuthRoute = pathname.startsWith(apiAuthPrefix);
  const isPublicRoute = publicRoutes.some((route) =>
    pathname.startsWith(route)
  );
  const isAuthRoute = authRoutes.some((route) => pathname.startsWith(route));

  if (isApiAuthRoute) {
    return response;
  }

  if (pathname === LOGIN_PAGE && isLoggedIn) {
    return NextResponse.redirect(new URL("/", request.url));
  }

  if (isAuthRoute) {
    if (!isLoggedIn) {
      const callbackUrl = encodeURIComponent(pathname);
      return NextResponse.redirect(
        new URL(
          `${LOGIN_PAGE}?callbackUrl=${callbackUrl}`,
          request.url
        )
      );
    }
    return response;
  }

  if (!isLoggedIn && !isPublicRoute) {
    return NextResponse.redirect(new URL(LOGIN_PAGE, request.url));
  }

  return response;
}

export const config = {
  matcher: ["/((?!.+\\.[\\w]+$|_next).*)", "/", "/(api|trpc)(.*)"],
};
