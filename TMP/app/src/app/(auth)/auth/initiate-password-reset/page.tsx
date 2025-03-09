"use client";

import Container from "@/components/container";
import InitiatePasswordResetForm from "@/features/auth/components/initiate-password-reset-form";

export default function InitiatePasswordResetPage() {
  return (
    <Container variant="center">
      <InitiatePasswordResetForm />
    </Container>
  );
}
