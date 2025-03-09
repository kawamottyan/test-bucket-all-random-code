import { Button } from "@/components/ui/button";
import { useAuthModalStore } from "@/stores/auth-modal-store";
import { useSession } from "next-auth/react";

interface AuthButtonProps {
  onClick: () => void;
  icon: React.ReactNode;
  variant:
    | "default"
    | "destructive"
    | "outline"
    | "secondary"
    | "ghost"
    | "link";
  size: "default" | "sm" | "lg" | "icon";
  className?: string;
  disabled?: boolean;
}

const AuthButton: React.FC<AuthButtonProps> = ({
  onClick,
  icon,
  variant,
  size,
  className,
  disabled,
}) => {
  const { data: session } = useSession();
  const { setLoginModalOpen } = useAuthModalStore();

  const handleClick = () => {
    if (!session) {
      setLoginModalOpen(true);
      return;
    }
    onClick();
  };

  return (
    <Button
      variant={variant}
      size={size}
      className={className}
      onClick={handleClick}
      disabled={disabled}
    >
      {icon}
    </Button>
  );
};

export default AuthButton;
