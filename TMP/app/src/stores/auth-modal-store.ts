import { create } from "zustand";

interface AuthModalState {
  isRegisterModalOpen: boolean;
  isLoginModalOpen: boolean;
  isLogoutModalOpen: boolean;
  setRegisterModalOpen: (isOpen: boolean) => void;
  setLoginModalOpen: (isOpen: boolean) => void;
  setLogoutModalOpen: (isOpen: boolean) => void;
}

export const useAuthModalStore = create<AuthModalState>((set) => ({
  isRegisterModalOpen: false,
  isLoginModalOpen: false,
  isLogoutModalOpen: false,
  setRegisterModalOpen: (isOpen) => set({ isRegisterModalOpen: isOpen }),
  setLoginModalOpen: (isOpen) => set({ isLoginModalOpen: isOpen }),
  setLogoutModalOpen: (isOpen) => set({ isLogoutModalOpen: isOpen }),
}));
