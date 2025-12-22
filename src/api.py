from fastapi import Header, HTTPException, Depends
from src.config import settings


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