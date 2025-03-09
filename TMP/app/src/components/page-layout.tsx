import * as React from "react";

const PageLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return <div className="mx-auto max-w-7xl rounded-lg">{children}</div>;
};

const PageHeader: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex flex-col space-y-1.5 p-6">{children}</div>
);

const PageTitle: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="text-2xl font-semibold leading-none tracking-tight">
    {children}
  </div>
);

const PageDescription: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <div className="text-sm text-muted-foreground">{children}</div>;

const PageBody: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="p-6 pt-0">{children}</div>
);

const PageFooter: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="flex w-full flex-col p-6 pt-0">{children}</div>
);

export {
  PageBody,
  PageDescription,
  PageFooter,
  PageHeader,
  PageLayout,
  PageTitle,
};
