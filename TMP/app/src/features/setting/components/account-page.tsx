import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertDialog, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import {
  linkSocialAccount,
  unlinkSocialAccount,
} from "@/features/auth/actions/social-account";
import { SafeCurrentUser } from "@/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { useQueryClient } from "@tanstack/react-query";
import { AlertCircle } from "lucide-react";
import { useSession } from "next-auth/react";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { SiGithub, SiGoogle } from "react-icons/si";
import { useAccountAction } from "../hooks/use-account-action";
import { AccountSchema } from "../schemas";
import { DeleteAccountModal } from "./delete-account-modal";
import EmailModal from "./email-modal";
import PasswordModal from "./password-modal";

interface AccountPageProps {
  user: SafeCurrentUser;
  isSuccess: boolean;
}

export default function AccountPage({ user, isSuccess }: AccountPageProps) {
  const { updateName, isPending: isUpdating } = useAccountAction();
  const [isGoogleAuth, setIsGoogleAuth] = useState(false);
  const [isGithubAuth, setIsGithubAuth] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | undefined>("");

  const [isEmailModalOpen, setIsEmailModalOpen] = useState(false);
  const [isPasswordModalOpen, setIsPasswordModalOpen] = useState(false);

  const isGoogleConnected = user.accounts?.some(
    (account) => account.provider === "google"
  );
  const isGithubConnected = user.accounts?.some(
    (account) => account.provider === "github"
  );

  const queryClient = useQueryClient();
  const { update } = useSession();

  const handleConnect = async (provider: "google" | "github") => {
    const setAuth = provider === "google" ? setIsGoogleAuth : setIsGithubAuth;
    const isConnected =
      provider === "google" ? isGoogleConnected : isGithubConnected;
    setAuth(true);

    try {
      if (isConnected) {
        const response = await unlinkSocialAccount(provider);
        if (!response.success) {
          setErrorMessage(response.message);
          return;
        }
        await queryClient.invalidateQueries({
          queryKey: CURRENT_USER_QUERY_KEY,
        });
        await update();
      } else {
        await linkSocialAccount(provider);
      }
    } catch {
      setErrorMessage(
        isConnected
          ? "Failed to disconnect social account. Please try again."
          : "Failed to connect social account. Please try again."
      );
    } finally {
      setAuth(false);
    }
  };

  const form = useForm({
    resolver: zodResolver(AccountSchema),
    defaultValues: {
      name: user.name ?? "",
    },
  });

  useEffect(() => {
    if (user) {
      form.reset({
        name: user.name ?? "",
      });
    }
  }, [user, form]);

  const onSubmit = form.handleSubmit((data) => {
    updateName(data.name);
  });

  return (
    <div className="space-y-6">
      <div className="mb-8 space-y-0.5 md:mb-0">
        <p className="text-lg font-bold tracking-tight">Account</p>
        <p className="text-sm text-muted-foreground">
          Update your account settings.
        </p>
      </div>
      {!isSuccess ? (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load user settings. Please try again later.
          </AlertDescription>
        </Alert>
      ) : (
        <Form {...form}>
          <form onSubmit={onSubmit}>
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input {...field} disabled={isUpdating} />
                  </FormControl>
                  <FormDescription>
                    This is your public display name.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="mt-6 flex justify-end">
              <Button type="submit" disabled={isUpdating}>
                {isUpdating ? "Updating..." : "Update"}
              </Button>
            </div>
          </form>
        </Form>
      )}
      <Separator className="my-8" />
      {errorMessage && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      )}
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">Email Address</p>
            <p className="text-sm text-muted-foreground">
              Change your email address to a new one
            </p>
          </div>
          <Button
            variant="outline"
            className="ml-4"
            onClick={() => setIsEmailModalOpen(true)}
          >
            Update
          </Button>
          <EmailModal
            email={user.email}
            open={isEmailModalOpen}
            onOpenChange={setIsEmailModalOpen}
          />
        </div>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">Password</p>
            <p className="text-sm text-muted-foreground">
              Update your password to keep your account secure
            </p>
          </div>
          <Button
            variant="outline"
            className="ml-4"
            onClick={() => setIsPasswordModalOpen(true)}
          >
            {user.hasPassword ? "Update" : "Set Password"}
          </Button>
          <PasswordModal
            hasPassword={user.hasPassword}
            open={isPasswordModalOpen}
            onOpenChange={setIsPasswordModalOpen}
          />
        </div>
      </div>
      <Separator className="my-8" />
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">Google</p>
            <p className="text-sm text-muted-foreground">
              Connect your account with Google for easier login
            </p>
          </div>
          <Button
            variant={isGoogleConnected ? "default" : "outline"}
            className="ml-4"
            onClick={() => handleConnect("google")}
            disabled={isGoogleAuth || isGithubAuth}
          >
            <SiGoogle className="mr-2 h-4 w-4" />
            {isGoogleAuth
              ? "Loading..."
              : isGoogleConnected
                ? "Connected"
                : "Connect"}
          </Button>
        </div>
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">GitHub</p>
            <p className="text-sm text-muted-foreground">
              Connect your account with GitHub for easier login
            </p>
          </div>
          <Button
            variant={isGithubConnected ? "default" : "outline"}
            className="ml-4"
            onClick={() => handleConnect("github")}
            disabled={isGoogleAuth || isGithubAuth}
          >
            <SiGithub className="mr-2 h-4 w-4" />
            {isGithubAuth
              ? "Loading..."
              : isGithubConnected
                ? "Connected"
                : "Connect"}
          </Button>
        </div>
      </div>
      <Separator className="my-8" />
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm font-medium">Delete Account</p>
            <p className="text-sm text-muted-foreground">
              Permanently delete your account and all associated data
            </p>
          </div>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" className="ml-4">
                Delete
              </Button>
            </AlertDialogTrigger>
            <DeleteAccountModal />
          </AlertDialog>
        </div>
      </div>
      <Separator className="my-8" />
    </div>
  );
}
