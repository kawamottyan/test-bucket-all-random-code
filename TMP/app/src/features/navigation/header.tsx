"use client";

import Logo from "@/components/logo";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { Commandbar } from "@/features/search/components/commandbar";
import { useCommandbar } from "@/features/search/hooks/use-commandbar";
import { useIsMobile } from "@/hooks/use-mobile";
import { Menu, PanelLeft, Search } from "lucide-react";
import { useSession } from "next-auth/react";
import NotificationSection from "../notification/components/notification-section";

const Header = () => {
  const { data: session } = useSession();
  const isMobile = useIsMobile();
  const { toggleSidebar, state } = useSidebar();
  const { setOpen: setCommandbarOpen } = useCommandbar();

  return (
    <header
      className={`fixed top-0 z-50 flex h-16 border-b bg-transparent bg-opacity-80 px-4 backdrop-blur duration-200 ease-linear md:px-8 ${
        !isMobile
          ? `${state === "expanded" ? "left-64 w-[calc(100%-16rem)]" : "left-0 w-full"}`
          : "left-0 w-full"
      }`}
    >
      <div className="flex w-full items-center justify-between">
        <div className="flex items-center md:gap-x-2">
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full bg-transparent hover:bg-muted"
            onClick={toggleSidebar}
          >
            <Menu className="h-5 w-5 md:hidden" />
            <PanelLeft className="hidden h-5 w-5 transition-transform duration-200 ease-linear group-hover:scale-105 md:block" />
          </Button>
          <Logo />
        </div>

        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <Button
            variant="outline"
            className="relative hidden h-8 w-80 select-none justify-start rounded bg-background text-sm font-normal text-muted-foreground lg:flex"
            onClick={() => setCommandbarOpen(true)}
          >
            <span className="inline-flex">Search</span>
            <kbd className="absolute right-2 select-none items-center rounded border border-muted-foreground bg-muted px-1.5">
              <span className="text-xs">/</span>
            </kbd>
          </Button>
        </div>

        <div className="flex items-center gap-x-1">
          <Button
            variant="secondary"
            size="icon"
            className="rounded-full bg-transparent hover:bg-muted lg:hidden"
            onClick={() => setCommandbarOpen(true)}
          >
            <Search className="h-5 w-5" />
          </Button>
          <Commandbar />
          {session && <NotificationSection />}
        </div>
      </div>
    </header>
  );
};

export default Header;
