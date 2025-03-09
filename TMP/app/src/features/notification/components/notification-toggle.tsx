import { Switch } from "@/components/ui/switch";
import { usePreferenceActions } from "@/features/setting/hooks/use-preference-actions";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { BellRing } from "lucide-react";

const NotificationToggle = () => {
  const { data: user } = useFetchCurrentUser();
  const { updateEmailNotification, isPending } = usePreferenceActions();

  return (
    <div className="flex items-center gap-x-4 rounded-md border p-4">
      <BellRing />
      <div className="flex-1">
        <p className="text-sm font-medium">Email Notification</p>
        <p className="text-sm text-muted-foreground">
          Receive notifications via email.
        </p>
      </div>
      <Switch
        checked={user?.allowEmailNotification ?? false}
        onCheckedChange={updateEmailNotification}
        disabled={isPending}
      />
    </div>
  );
};

export default NotificationToggle;
