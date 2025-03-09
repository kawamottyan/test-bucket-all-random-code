"use client";

import LoginModal from "@/features/auth/components/login-modal";
import { LogoutModal } from "@/features/auth/components/logout-modal";
import RegisterModal from "@/features/auth/components/register-modal";
import { useAuthModalStore } from "@/stores/auth-modal-store";

export function ModalProvider() {
  const {
    isRegisterModalOpen,
    setRegisterModalOpen,
    isLoginModalOpen,
    setLoginModalOpen,
    isLogoutModalOpen,
    setLogoutModalOpen,
  } = useAuthModalStore();
  return (
    <>
      <LoginModal open={isLoginModalOpen} onOpenChange={setLoginModalOpen} />
      <RegisterModal
        open={isRegisterModalOpen}
        onOpenChange={setRegisterModalOpen}
      />
      <LogoutModal open={isLogoutModalOpen} onOpenChange={setLogoutModalOpen} />
    </>
  );
}
