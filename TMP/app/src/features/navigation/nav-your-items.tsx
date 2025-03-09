import {
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { Bookmark, Star } from "lucide-react";
import Link from "next/link";

export function NavYourItems() {
  const { data: currentUser, isPending } = useFetchCurrentUser();
  const { setOpenMobile } = useSidebar();
  const handleLinkClick = () => {
    setOpenMobile(false);
  };

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton asChild>
          <Link href="/bookmark" onClick={handleLinkClick}>
            <Bookmark />
            <span>Bookmark</span>
            {!isPending && (
              <SidebarMenuBadge>{currentUser?.bookmarkCount}</SidebarMenuBadge>
            )}
          </Link>
        </SidebarMenuButton>
      </SidebarMenuItem>
      <SidebarMenuItem>
        <SidebarMenuButton asChild>
          <Link href="/review" onClick={handleLinkClick}>
            <Star />
            <span>Review</span>
            {!isPending && (
              <SidebarMenuBadge>{currentUser?.reviewCount}</SidebarMenuBadge>
            )}
          </Link>
        </SidebarMenuButton>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
