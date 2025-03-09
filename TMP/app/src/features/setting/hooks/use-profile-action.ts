import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { toast } from "sonner";

export const useProfileAction = () => {
  const queryClient = useQueryClient();
  const { data: user } = useFetchCurrentUser();
  const { update: updateSession } = useSession();

  const update = useMutation({
    mutationFn: async (username: string) => {
      if (!user?.id) {
        throw new Error("User not authorized");
      }
      const response = await fetch(`/api/profile/${user.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username }),
      });

      if (!response.ok) {
        if (response.status === 409) {
          throw new Error("Username already exists");
        }
        throw new Error("Failed to update profile");
      }

      return response.json();
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      await updateSession();
      toast.success("Profile updated successfully");
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  return {
    isPending: update.isPending,
    updateUsername: (username: string) => update.mutate(username),
  };
};
