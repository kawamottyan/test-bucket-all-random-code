import { formatDistanceToNowStrict } from "date-fns";
import { enUS } from "date-fns/locale";

type FormatDistanceToken = {
  [key: string]: string;
};

interface FormatDistanceOptions {
  addSuffix?: boolean;
  comparison?: number;
}

const formatDistanceLocale: Readonly<FormatDistanceToken> = {
  lessThanXSeconds: "now",
  xSeconds: "now",
  halfAMinute: "now",
  lessThanXMinutes: "{{count}}m",
  xMinutes: "{{count}}m",
  aboutXHours: "{{count}}h",
  xHours: "{{count}}h",
  xDays: "{{count}}d",
  aboutXWeeks: "{{count}}w",
  xWeeks: "{{count}}w",
  aboutXMonths: "{{count}} month",
  xMonths: "{{count}} month",
  aboutXYears: "{{count}} year",
  xYears: "{{count}} year",
  overXYears: "{{count}} year",
  almostXYears: "{{count}} year",
};

function formatDistance(
  token: string,
  count: number,
  options: FormatDistanceOptions = {}
): string {
  const result = formatDistanceLocale[
    token as keyof typeof formatDistanceLocale
  ].replace("{{count}}", count.toString());

  if (options.addSuffix) {
    if (options.comparison && options.comparison > 0) {
      return "in " + result;
    } else {
      if (result === "now") return result;
      return result;
    }
  }

  return result;
}

export function formatDate(startDate: string): string {
  try {
    const date = new Date(startDate);
    if (isNaN(date.getTime())) {
      throw new Error("Invalid date");
    }

    return new Intl.DateTimeFormat("en-US", {
      month: "long",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "numeric",
      hour12: true,
    })
      .format(date)
      .replace(" at ", ", ");
  } catch (error) {
    throw new Error(
      `Failed to format date: ${error instanceof Error ? error.message : "Unknown error"}`
    );
  }
}

export function formatTimeToNow(date: Date, appendSuffix?: boolean): string {
  const formattedTime = formatDistanceToNowStrict(date, {
    addSuffix: true,
    locale: {
      ...enUS,
      formatDistance,
    },
  });

  if (appendSuffix && formattedTime !== "now") {
    return `${formattedTime} ago`;
  }

  return formattedTime;
}

export const truncateText = (text: string, maxLength: number): string => {
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
};

export const formatIsoDate = (date: Date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
};

export const formatIsoDateInput = (input: string): string => {
  const nums = input.replace(/[^\d-]/g, "");
  const match = nums.match(/^(\d{0,4})-?(\d{0,2})-?(\d{0,2})$/);
  if (!match) return nums;

  const [, year, month, day] = match;

  if (!year) return "";
  if (!month) return year;
  if (!day) return `${year}-${month.padStart(2, "0")}`;

  return `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`;
};
