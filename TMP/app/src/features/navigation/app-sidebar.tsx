"use client";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
} from "@/components/ui/sidebar";
import { useSession } from "next-auth/react";
import NavLegal from "./nav-legal";
import { NavSupport } from "./nav-support";
import { NavUser } from "./nav-user";
import { NavYourItems } from "./nav-your-items";

export function AppSidebar() {
  const { data: session } = useSession();
  return (
    <Sidebar>
      <SidebarHeader className="h-16 border-b border-sidebar-border">
        <NavUser />
      </SidebarHeader>
      <SidebarContent>
        {session && (
          <SidebarGroup>
            <SidebarGroupLabel>You</SidebarGroupLabel>
            <SidebarGroupContent>
              <NavYourItems />
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        <SidebarGroup>
          <SidebarGroupLabel>Support</SidebarGroupLabel>
          <SidebarGroupContent>
            <NavSupport />
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <NavLegal />
      </SidebarFooter>
    </Sidebar>
  );
}
