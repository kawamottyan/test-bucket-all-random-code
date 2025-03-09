import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { useState } from "react";
import { toast } from "sonner";
import { MAX_FILE_SIZE } from "../constants";

interface UseImageUploadReturn {
  uploadImage: (file: File) => Promise<void>;
  removeImage: () => Promise<void>;
  isUploading: boolean;
  error: string | null;
}

export function useImageUpload(userId: string): UseImageUploadReturn {
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { update: updateSession } = useSession();

  const uploadImage = async (file: File) => {
    try {
      setIsUploading(true);
      setError(null);

      if (file.size > MAX_FILE_SIZE) {
        throw new Error("File size exceeds 5MB limit");
      }

      const response = await fetch("/api/profiles/image/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          filename: file.name,
          contentType: file.type,
          fileSize: file.size,
          userId: userId,
        }),
      });

      const responseData = await response.json();

      if (!response.ok) {
        throw new Error(responseData.message || "Failed to get upload URL");
      }

      const { data } = responseData;
      const { url, imageUrl } = data;
      const uploadResponse = await fetch(url, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": file.type,
        },
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload image to R2");
      }

      const updateResponse = await fetch(`/api/profiles/${userId}/image/`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: imageUrl,
        }),
      });

      if (!updateResponse.ok) {
        const errorData = await updateResponse.json();
        throw new Error(errorData.message || "Failed to update profile");
      }

      await queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      await updateSession();
      toast.success("Profile image updated successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload image");
      throw err;
    } finally {
      setIsUploading(false);
    }
  };
  const removeImage = async () => {
    try {
      setIsUploading(true);
      setError(null);

      const response = await fetch(`/api/profiles/${userId}/image/`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: null,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to remove image");
      }

      await queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      await updateSession();
      toast.success("Profile image removed successfully");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove image");
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  return {
    uploadImage,
    removeImage,
    isUploading,
    error,
  };
}
