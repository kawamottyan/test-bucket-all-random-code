import { CURRENT_USER_QUERY_KEY } from "@/constants/query-keys";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { EmailFrequency } from "@prisma/client";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { PreferenceFormValues } from "../schemas";

export const usePreferenceActions = () => {
  "use memo";

  const queryClient = useQueryClient();
  const { data: user } = useFetchCurrentUser();

  const update = useMutation({
    mutationFn: async (data: Partial<PreferenceFormValues>) => {
      if (!user?.id) {
        throw new Error("User not authorized");
      }

      const response = await fetch(`/api/preferences/${user.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error("Failed to update preference");
      }

      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: CURRENT_USER_QUERY_KEY });
      toast.success("Notification settings updated");
    },
    onError: () => {
      toast.error("Failed to update notification settings");
    },
  });

  const updateEmailNotification = (enabled: boolean) => {
    update.mutate({
      allowEmailNotification: enabled,
      ...(!enabled && { emailFrequency: EmailFrequency.NONE }),
    });
  };

  const updatePreference = (data: Partial<PreferenceFormValues>) => {
    update.mutate(data);
  };

  return {
    updateEmailNotification,
    updatePreference,
    isPending: update.isPending,
  };
};
