import { getUserById } from "@/actions/get-user";
import { db } from "@/lib/db";
import { generateUniqueUsername } from "@/lib/username";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { UserRole } from "@prisma/client";
import NextAuth from "next-auth";
import authConfig from "./auth.config";

export const { handlers, signIn, signOut, auth } = NextAuth({
  pages: {
    signIn: '/auth/login',
    error: "/auth/error",
  },
  events: {
    async createUser({ user }) {
      const username = await generateUniqueUsername();

      await db.user.update({
        where: { id: user.id },
        data: { username },
      });
    },
    async linkAccount({ user }) {
      await db.user.update({
        where: { id: user.id },
        data: { emailVerified: new Date() },
      });
    },
  },
  callbacks: {
    async signIn({ user, account }) {
      if (account?.provider !== "credentials") return true;

      if (!user.id) return false;
      const existingUser = await getUserById(user.id);
      if (!existingUser?.emailVerified) return false;

      return true;
    },

    async session({ token, session }) {
      if (token.sub && session.user) {
        session.user.id = token.sub;
      }

      if (token.name && session.user) {
        session.user.name = token.name as string;
      }

      if (token.email && session.user) {
        session.user.email = token.email as string;
      }

      if (token.image && session.user) {
        session.user.image = token.image as string | null;
      }

      if (token.username && session.user) {
        session.user.username = token.username as string | null;
      }

      if (token.role && session.user) {
        session.user.role = token.role as UserRole;
      }

      return session;
    },
    async jwt({ token }) {
      if (!token.sub) return token;

      const existingUser = await getUserById(token.sub);
      if (!existingUser) return token;

      token.name = existingUser.name;
      token.email = existingUser.email;
      token.image = existingUser.image;
      token.username = existingUser.username;
      token.role = existingUser.role;
      return token;
    },
  },
  adapter: PrismaAdapter(db),
  session: { strategy: "jwt" },
  ...authConfig,
});
