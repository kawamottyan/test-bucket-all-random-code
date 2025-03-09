import React from "react";

const NotificationIndicator: React.FC = () => (
  <span className="absolute right-1 top-1 flex h-2 w-2">
    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-destructive opacity-75" />
    <span className="relative inline-flex h-2 w-2 rounded-full bg-destructive" />
  </span>
);

export default NotificationIndicator;
