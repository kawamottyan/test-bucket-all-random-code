import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

export function useDeleteAccount() {
  const router = useRouter();
  const { data: user } = useFetchCurrentUser();
  const queryClient = useQueryClient();
  const { update: updateSession } = useSession();

  return useMutation({
    mutationFn: async () => {
      if (!user?.id) {
        throw new Error("User not authorized");
      }
      const response = await fetch(`/api/accounts/${user.id}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error("Failed to delete account");
      }

      return response.json();
    },
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      await updateSession();
      router.push("/");
      toast.success("Your account has been successfully deleted");
    },
    onError: () => {
      toast.error("Failed to delete account. Please try again.");
    },
  });
}
