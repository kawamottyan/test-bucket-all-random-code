"use client";

import { ChevronsUpDown, LogIn, LogOut, Settings } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { LoginModal } from "@/features/auth/components/login-modal";
import { LogoutModal } from "@/features/auth/components/logout-modal";
import { useAuthModalStore } from "@/stores/auth-modal-store";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";

export function NavUser() {
  const { data: session } = useSession();
  const {
    isLoginModalOpen,
    isLogoutModalOpen,
    setLoginModalOpen,
    setLogoutModalOpen,
  } = useAuthModalStore();
  const { isMobile, setOpenMobile } = useSidebar();
  const router = useRouter();

  const handleLoginClick = () => {
    setLoginModalOpen(true);
    setOpenMobile(false);
  };

  const handleSettingClick = () => {
    router.push("/setting");
    setOpenMobile(false);
  };

  if (!session?.user) {
    return (
      <SidebarMenu className="flex h-full flex-col justify-center py-2">
        <SidebarMenuItem>
          <Button
            className="flex w-full items-center justify-center gap-2"
            onClick={handleLoginClick}
          >
            <LogIn className="h-4 w-4" />
            <span>Sign In to MovieAI</span>
          </Button>
        </SidebarMenuItem>
        <LoginModal open={isLoginModalOpen} onOpenChange={setLoginModalOpen} />
      </SidebarMenu>
    );
  }

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size="lg"
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            >
              <Avatar
                key={session.user.image || "fallback"}
                className="h-8 w-8"
              >
                {session.user.image ? (
                  <AvatarImage
                    src={session.user.image}
                    alt={session.user.username || "User avatar"}
                  />
                ) : (
                  <AvatarFallback>
                    {(session.user.name ?? "").slice(0, 2).toUpperCase() || "U"}
                  </AvatarFallback>
                )}
              </Avatar>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">
                  {session.user.name}
                </span>
                <span className="truncate text-xs">
                  {`@${session.user?.username ?? "unknown"}`}
                </span>
              </div>
              <ChevronsUpDown className="ml-auto size-4" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg"
            side={isMobile ? "bottom" : "right"}
            align="start"
            sideOffset={4}
          >
            <DropdownMenuGroup>
              <DropdownMenuItem onClick={handleSettingClick}>
                <Settings />
                Setting
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => setLogoutModalOpen(true)}>
              <LogOut />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
      <LogoutModal open={isLogoutModalOpen} onOpenChange={setLogoutModalOpen} />
    </SidebarMenu>
  );
}
