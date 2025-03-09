import { useSidebar } from "@/components/ui/sidebar";
import Link from "next/link";

const NavLegal = () => {
  const { setOpenMobile } = useSidebar();
  const handleLinkClick = () => {
    setOpenMobile(false);
  };
  return (
    <div className="flex flex-col items-start gap-y-2 p-4 text-xs">
      <Link
        href="/privacy-policy"
        className="hover:underline"
        onClick={handleLinkClick}
      >
        Privacy Policy
      </Link>
      <Link
        href="/terms-of-service"
        className="hover:underline"
        onClick={handleLinkClick}
      >
        Terms of Service
      </Link>
      <Link
        href="/community"
        className="hover:underline"
        onClick={handleLinkClick}
      >
        Community
      </Link>
      <Link
        href="/contact"
        className="hover:underline"
        onClick={handleLinkClick}
      >
        Contact
      </Link>
    </div>
  );
};

export default NavLegal;
