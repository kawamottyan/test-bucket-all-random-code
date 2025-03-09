const AUTH_MESSAGES: Record<string, {
  title: string;
  description: string;
  variant?: "default" | "destructive"
}> = {
  passworRreset: {
    title: "Password Reset Success",
    description: "Your password has been reset successfully. Please log in with your new password",
    variant: "default"
  },
  emailVerified: {
    title: "Email Verified",
    description: "Your email has been verified successfully. You can now log in to your account",
    variant: "default"
  },
  OAuthAccountNotLinked: {
    title: "Account Link Error",
    description: "Email already in use with different provider",
    variant: "destructive"
  },
  OAuthSignInError: {
    title: "Sign In Error",
    description: "Failed to sign in with social account",
    variant: "destructive"
  },
  CredentialsSignin: {
    title: "Sign In Error",
    description: "Invalid email or password",
    variant: "destructive"
  },
  SocialSignInError: {
    title: "Sign In Error",
    description: "Failed to login with social account. Please try again",
    variant: "destructive"
  },
  Default: {
    title: "Error Occurred",
    description: "An unexpected error occurred. Please try again later",
    variant: "destructive"
  }
};

export const getAuthMessage = (code: string | null) => {
  if (!code) return null;
  return AUTH_MESSAGES[code] || AUTH_MESSAGES.Default;
};