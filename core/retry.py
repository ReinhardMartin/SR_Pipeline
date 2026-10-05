import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from openai import APIConnectionError, APIStatusError


def retry_call(operation, *, retries=2, initial=1.0, maximum=60.0):
    for attempt in range(retries + 1):
        try:
            return operation()
        except (APIConnectionError, APIStatusError, httpx.TransportError, httpx.HTTPStatusError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if attempt == retries or status is not None and status not in (408, 409, 429) and status < 500:
                raise
            delay = random.uniform(0, min(maximum, initial * 2 ** attempt))
            if response is not None:
                header = response.headers.get("retry-after")
                if header:
                    try:
                        requested = float(header)
                    except ValueError:
                        try:
                            requested = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
                        except (TypeError, ValueError, OverflowError):
                            requested = 0
                    if requested > maximum:
                        raise
                    delay = max(delay, requested)
            time.sleep(delay)
