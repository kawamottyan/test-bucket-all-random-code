"use client";

import Container from "@/components/container";
import {
  PageBody,
  PageDescription,
  PageHeader,
  PageLayout,
  PageTitle,
} from "@/components/page-layout";
import { Button } from "@/components/ui/button";
import AccountPage from "@/features/setting/components/account-page";
import PreferencePage from "@/features/setting/components/preference-page";
import ProfilePage from "@/features/setting/components/profile-page";
import SettingLoadingPage from "@/features/setting/components/setting-loading";
import { sidebarItems, TabId } from "@/features/setting/constants";
import { useFetchCurrentUser } from "@/hooks/use-fetch-current-user";
import { useState } from "react";

export default function SettingPage() {
  const { data: user, isPending, isSuccess } = useFetchCurrentUser();
  const [activeTab, setActiveTab] = useState<TabId>("account");

  if (!user) {
    return null;
  }

  if (isPending) {
    return (
      <Container variant="center">
        <div className="grid grid-cols-1 md:grid-cols-4">
          <nav className="mt-4 flex md:col-span-1 md:flex-col">
            {sidebarItems.map((item) => (
              <Button
                key={item.id}
                variant="ghost"
                disabled
                className="justify-start rounded-md px-4 py-2 md:w-full"
              >
                {item.title}
              </Button>
            ))}
          </nav>
          <div className="flex flex-col md:col-span-3">
            <SettingLoadingPage />
          </div>
        </div>
      </Container>
    );
  }

  return (
    <Container variant="topmargin">
      <PageLayout>
        <PageHeader>
          <PageTitle>Settings</PageTitle>
          <PageDescription>
            Manage your account preferences and customize your experience.
          </PageDescription>
        </PageHeader>
        <PageBody>
          <div className="grid grid-cols-1 md:grid-cols-4">
            <nav className="mb-4 flex md:col-span-1 md:flex-col md:pr-4">
              {sidebarItems.map((item) => (
                <Button
                  key={item.id}
                  variant="ghost"
                  onClick={() => setActiveTab(item.id)}
                  className={`${
                    activeTab === item.id ? "bg-accent" : ""
                  } justify-start rounded-md px-4 py-2 md:w-full`}
                >
                  {item.title}
                </Button>
              ))}
            </nav>
            <div className="flex flex-col md:col-span-3">
              {activeTab === "account" && (
                <AccountPage user={user} isSuccess={isSuccess} />
              )}
              {activeTab === "profile" && (
                <ProfilePage user={user} isSuccess={isSuccess} />
              )}
              {activeTab === "preference" && (
                <PreferencePage user={user} isSuccess={isSuccess} />
              )}
            </div>
          </div>
        </PageBody>
      </PageLayout>
    </Container>
  );
}
