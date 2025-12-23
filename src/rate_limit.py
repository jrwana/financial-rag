import time
from collections import defaultdict
from fastapi import Request, HTTPException

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def _get_key(self, request: Request) -> str:
        """Extract identifier: API key if present, else IP"""
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        # fallback to IP (handle X-Forwarded-For for proxies)
        forwarded = request.headers.get("X-Forwarded-For")
        ip = forwarded.split(",")[0].strip() if forwarded else request.client.host
        return f"ip:{ip}"

    def check(self, request: Request) -> None:
        """Raise 429 if rate limit exceeded"""
        key = self._get_key(request)
        now = time.time()
        window_start = now - 60

        # Clean old requests outside the window
        self.requests[key] = [t for t in self.requests[key] if t > window_start]

        if len(self.requests[key]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.requests_per_minute} requests/minute. Try again shortly."
            )

        self.requests[key].append(now)


# Singleton instance
limiter = RateLimiter(requests_per_minute=60)