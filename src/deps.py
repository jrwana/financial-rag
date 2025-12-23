from fastapi import Header, HTTPException, Depends, Request
from .config import settings
from .rate_limit import limiter


def require_api_key(x_api_key: str | None = Header(None)):
    if settings.ENV != "prod":
        return # local mode, skip auth

    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")

    if x_api_key != settings.API_KEY:
        raise HTTPException(403, "Invalid API key")


def require_admin_key(x_admin_key: str | None = Header(None)):
    if settings.ENV != "prod":
        return # local mode, skip auth

    if not x_admin_key:
        raise HTTPException(401, "Missing X-Admin-Key header")

    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(403, "Invalid Admin key")


async def check_rate_limit(request: Request) -> None:
    """Rate limit check - only in prod"""
    if settings.ENV != "prod":
        return
    limiter.check(request)