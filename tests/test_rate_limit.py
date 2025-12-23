import pytest
from unittest.mock import Mock
from src.rate_limit import RateLimiter
from fastapi import HTTPException

def test_allows_requests_within_limit():
    limiter = RateLimiter(requests_per_minute=5)
    request = Mock()
    request.headers = {"X-API-Key": "test-key"}
    request.client.host = "127.0.0.1"

    # Should allow 5 requests
    for _ in range(5):
        limiter.check(request)  # No exception

def test_blocks_after_limit_exceeded():
    limiter = RateLimiter(requests_per_minute=3)
    request = Mock()
    request.headers = {"X-API-Key": "test-key"}
    request.client.host = "127.0.0.1"

    for _ in range(3):
        limiter.check(request)

    with pytest.raises(HTTPException) as exc:
        limiter.check(request)
    assert exc.value.status_code == 429

def test_different_keys_have_separate_limits():
    limiter = RateLimiter(requests_per_minute=2)

    req1 = Mock()
    req1.headers = {"X-API-Key": "key-1"}
    req1.client.host = "127.0.0.1"

    req2 = Mock()
    req2.headers = {"X-API-Key": "key-2"}
    req2.client.host = "127.0.0.1"

    # Both keys should get their own limit
    for _ in range(2):
        limiter.check(req1)
        limiter.check(req2)

    # key-1 blocked, key-2 should also be blocked
    with pytest.raises(HTTPException):
        limiter.check(req1)