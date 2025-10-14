#!/usr/bin/env python3
"""
Wait for the streaming backend to become reachable before invoking record_stream.py.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Sequence

import requests

DEFAULT_WAIT_URL = "http://backend:8000/health"
DEFAULT_WAIT_TIMEOUT = 120.0
DEFAULT_WAIT_INTERVAL = 2.0


def wait_for_backend(
    url: str,
    timeout: float,
    interval: float,
) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code < 500:
                return
        except Exception:
            pass

        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for backend at {url}")

        time.sleep(interval)


def main(argv: Sequence[str]) -> int:
    wait_url = os.getenv("STREAM_WAIT_URL", DEFAULT_WAIT_URL)
    wait_timeout = float(os.getenv("STREAM_WAIT_TIMEOUT", DEFAULT_WAIT_TIMEOUT))
    wait_interval = float(os.getenv("STREAM_WAIT_INTERVAL", DEFAULT_WAIT_INTERVAL))

    print(f"[wait] Waiting for backend: {wait_url}", flush=True)
    try:
        wait_for_backend(wait_url, wait_timeout, wait_interval)
    except TimeoutError as exc:
        print(f"[wait] {exc}", file=sys.stderr, flush=True)
        return 1

    print("[wait] Backend ready, starting recorder...", flush=True)
    args = ["python", "/app/record_stream.py", *argv]
    os.execvp(args[0], args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
