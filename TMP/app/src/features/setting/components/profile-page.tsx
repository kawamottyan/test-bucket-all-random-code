import { Alert, AlertDescription } from "@/components/ui/alert";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
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
import { SafeCurrentUser } from "@/types";
import { zodResolver } from "@hookform/resolvers/zod";
import * as React from "react";
import { useEffect, useRef } from "react";
import { useForm } from "react-hook-form";
import { useImageUpload } from "../hooks/use-image-upload";
import { useProfileAction } from "../hooks/use-profile-action";
import { ProfileSchema } from "../schemas";

interface ProfilePageProps {
  user: SafeCurrentUser;
  isSuccess: boolean;
}

const ProfilePage: React.FC<ProfilePageProps> = ({ user, isSuccess }) => {
  const { updateUsername, isPending: isUpdating } = useProfileAction();
  const { uploadImage, removeImage, isUploading } = useImageUpload(user.id);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const form = useForm({
    resolver: zodResolver(ProfileSchema),
    defaultValues: {
      username: user.username ?? "",
    },
  });

  useEffect(() => {
    if (user) {
      form.reset({
        username: user.username ?? "",
      });
    }
  }, [user, form]);

  const onSubmit = form.handleSubmit((data) => {
    updateUsername(data.username);
  });

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      await uploadImage(file);
    } catch (error) {
      console.log("Failed to upload image", error);
    }
  };

  const handleRemoveImage = async () => {
    try {
      await removeImage();
    } catch (error) {
      console.log("Failed to remove image", error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="mb-8 space-y-0.5 md:mb-0">
        <p className="text-lg font-bold tracking-tight">Profile</p>
        <p className="text-sm text-muted-foreground">
          Update your profile settings.
        </p>
      </div>
      {!isSuccess ? (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load user settings. Please try again later.
          </AlertDescription>
        </Alert>
      ) : (
        <>
          <Form {...form}>
            <form onSubmit={onSubmit}>
              <FormField
                control={form.control}
                name="username"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>User Name</FormLabel>
                    <FormControl>
                      <Input {...field} disabled={isUpdating} />
                    </FormControl>
                    <FormDescription>
                      Your unique identifier in this platform.
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
          <Separator className="my-8" />
          <div>
            <div className="text-sm font-medium">Profile Image</div>
            <div className="mb-4 text-sm text-muted-foreground">
              Upload a profile picture to make your account more personal.
            </div>
            <Input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept="image/jpeg,image/png,image/gif,image/webp"
              onChange={handleFileChange}
            />

            <div className="mt-2 flex items-center justify-between">
              <Avatar key={user.image || "fallback"} className="h-16 w-16">
                {user.image ? (
                  <AvatarImage
                    src={user.image}
                    alt={user.username || "User avatar"}
                  />
                ) : (
                  <AvatarFallback>
                    {(user.name ?? "").slice(0, 2).toUpperCase() || "U"}
                  </AvatarFallback>
                )}
              </Avatar>

              <div className="flex gap-2">
                <Button
                  type="button"
                  size="sm"
                  className="flex items-center gap-2"
                  onClick={handleUploadClick}
                  disabled={isUploading}
                >
                  {isUploading ? "Uploading..." : "Upload"}
                </Button>
                {user.image && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2"
                    onClick={handleRemoveImage}
                    disabled={isUploading}
                  >
                    Remove
                  </Button>
                )}
              </div>
            </div>
          </div>
          <Separator className="my-8" />
        </>
      )}
    </div>
  );
};

export default ProfilePage;
