import { Alert, AlertDescription } from "@/components/ui/alert";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { Separator } from "@/components/ui/separator";
import { SafeCurrentUser } from "@/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { EmailFrequency } from "@prisma/client";
import { useEffect } from "react";
import { useForm } from "react-hook-form";
import { usePreferenceActions } from "../hooks/use-preference-actions";
import { PreferenceFormValues, PreferenceSchema } from "../schemas";

interface PreferencePageProps {
  user: SafeCurrentUser;
  isSuccess: boolean;
}

export default function PreferencePage({
  user,
  isSuccess,
}: PreferencePageProps) {
  const { updatePreference, isPending: isUpdateing } = usePreferenceActions();

  const form = useForm<PreferenceFormValues>({
    resolver: zodResolver(PreferenceSchema),
    defaultValues: {
      allowEmailNotification: user?.allowEmailNotification ?? false,
      emailFrequency: user?.emailFrequency ?? EmailFrequency.NONE,
    },
  });

  useEffect(() => {
    if (user) {
      form.reset({
        allowEmailNotification: user.allowEmailNotification,
        emailFrequency: user.emailFrequency,
      });
    }
  }, [user, form]);

  const onSubmit = form.handleSubmit((data) => {
    updatePreference(data);
  });

  return (
    <div className="space-y-6">
      <div className="mb-8 space-y-0.5 md:mb-0">
        <p className="text-lg font-bold tracking-tight">Preferences</p>
        <p className="text-sm text-muted-foreground">
          Update your preference settings.
        </p>
      </div>
      {!isSuccess ? (
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load user preferences. Please try again.
          </AlertDescription>
        </Alert>
      ) : (
        <Form {...form}>
          <form onSubmit={onSubmit} className="space-y-6">
            <FormField
              control={form.control}
              name="allowEmailNotification"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email Notifications</FormLabel>
                  <FormControl>
                    <Select
                      onValueChange={(value) => {
                        const boolValue = value === "true";
                        field.onChange(boolValue);
                      }}
                      value={String(field.value)}
                      disabled={isUpdateing}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select notification setting" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="true">Enabled</SelectItem>
                        <SelectItem value="false">Disabled</SelectItem>
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormDescription>
                    Choose whether to receive email notifications about your
                    account and activity.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="emailFrequency"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email Frequency</FormLabel>
                  <FormControl>
                    <Select
                      onValueChange={field.onChange}
                      value={field.value}
                      disabled={isUpdateing}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select email frequency" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.values(EmailFrequency).map((freq) => (
                          <SelectItem key={freq} value={freq}>
                            {freq.charAt(0) + freq.slice(1).toLowerCase()}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </FormControl>
                  <FormDescription>
                    Control how often you receive updates, such as movie
                    recommendations. Select &apos;None&apos; to disable regular
                    updates.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <div className="flex justify-end">
              <Button type="submit" disabled={isUpdateing}>
                {isUpdateing ? "Updating..." : "Update"}
              </Button>
            </div>
          </form>
        </Form>
      )}
      <Separator className="my-8" />
    </div>
  );
}
