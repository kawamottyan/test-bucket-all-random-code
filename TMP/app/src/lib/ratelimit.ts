import redis from "@/db/redis";
import { Ratelimit } from "@upstash/ratelimit";

const ratelimit = new Ratelimit({
  redis: redis,
  limiter: Ratelimit.slidingWindow(100, "5m"),
  prefix: "@upstash/ratelimit",
  analytics: false,
});

export default ratelimit;
