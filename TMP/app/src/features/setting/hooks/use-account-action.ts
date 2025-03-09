import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { toast } from "sonner";

export const useAccountAction = () => {
  const queryClient = useQueryClient();
  const { data: user } = useFetchCurrentUser();
  const { update: updateSession } = useSession();

  const update = useMutation({
    mutationFn: async (name: string) => {
      if (!user?.id) {
        throw new Error("User not authorized");
      }
      const response = await fetch(`/api/accounts/${user.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name }),
      });

      if (!response.ok) {
        throw new Error("Failed to update account");
      }

      return response.json();
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      await updateSession();
      toast.success("Account updated successfully");
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  return {
    isPending: update.isPending,
    updateName: (name: string) => update.mutate(name),
  };
};
