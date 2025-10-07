#!/usr/bin/env python3
"""
Record the MJPEG camera stream to a video file.

This utility connects to the FastAPI backend's ``/stream.mjpg`` endpoint,
decodes each JPEG frame, and writes the sequence to an MP4 file using OpenCV.

Example:
    python record_stream.py --url http://localhost:8000/stream.mjpg --output capture.mp4
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from dataclasses import dataclass
from typing import Iterable, Optional
import time

import cv2
import numpy as np
import requests

LOGGER = logging.getLogger("record_stream")


@dataclass
class FramePacket:
    data: bytes
    timestamp: Optional[float] = None


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/stream.mjpg",
        help="URL to the MJPEG stream (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="recording.mp4",
        help="Filepath for the recorded video (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video (default: %(default)s)",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=0,
        help="Optional limit on number of frames to record (0 = unlimited).",
    )
    parser.add_argument(
        "--boundary",
        default="frame",
        help=(
            "Multipart boundary token if the server uses a non-default value "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def iter_mjpeg_frames(response: requests.Response, boundary: str) -> Iterable[FramePacket]:
    boundary_bytes = f"--{boundary}".encode()
    closing_boundary = boundary_bytes + b"--"
    buffer = bytearray()

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buffer.extend(chunk)

        while True:
            boundary_index = buffer.find(boundary_bytes)
            if boundary_index == -1:
                break

            if boundary_index > 0:
                del buffer[:boundary_index]
                boundary_index = 0

            if buffer.startswith(closing_boundary):
                return

            after_boundary = boundary_index + len(boundary_bytes)
            if len(buffer) < after_boundary + 2:
                break

            if buffer[after_boundary:after_boundary + 2] == b"\r\n":
                header_start = after_boundary + 2
            else:
                header_start = after_boundary

            header_end = buffer.find(b"\r\n\r\n", header_start)
            if header_end == -1:
                break

            header_bytes = buffer[header_start:header_end]
            headers = _parse_headers(header_bytes)

            try:
                content_length = int(headers.get("content-length", "0"))
            except ValueError:
                content_length = 0

            body_start = header_end + 4
            frame_end = body_start + content_length
            trailer_end = frame_end + 2  # Expect trailing CRLF.

            available = len(buffer) - body_start
            if content_length <= 0 or len(buffer) < trailer_end:
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "Waiting for frame payload: have %d bytes, need %d",
                        max(available, 0),
                        content_length,
                    )
                break

            frame_bytes = bytes(buffer[body_start:frame_end])
            del buffer[:trailer_end]

            LOGGER.debug(
                "Extracted frame chunk len=%d (content-length=%d)",
                len(frame_bytes),
                content_length,
            )

            yield FramePacket(data=frame_bytes)


def _parse_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw.split(b"\r\n"):
        if b":" not in line:
            continue
        key, value = line.split(b":", 1)
        headers[key.strip().lower().decode()] = value.strip().decode()
    return headers


def decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
    array = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


def record_stream(args: argparse.Namespace) -> None:
    LOGGER.info("Connecting to %s", args.url)
    response = requests.get(args.url, stream=True, timeout=10)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    boundary = args.boundary
    if "boundary=" in content_type:
        boundary = content_type.split("boundary=")[-1].strip()
        LOGGER.debug("Detected boundary from headers: %s", boundary)

    writer: Optional[cv2.VideoWriter] = None
    frame_count = 0
    last_log_time = time.time()

    try:
        for packet in iter_mjpeg_frames(response, boundary):
            frame = decode_frame(packet.data)
            if frame is None:
                LOGGER.warning("Failed to decode frame %d", frame_count)
                continue

            if writer is None:
                height, width = frame.shape[:2]
                LOGGER.info("Opening VideoWriter with resolution %dx%d", width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Failed to open VideoWriter for output.")
            else:
                LOGGER.debug(
                    "Decoded frame %d (%dx%d)", frame_count + 1, frame.shape[1], frame.shape[0]
                )

            writer.write(frame)
            frame_count += 1
            last_log_time = time.time()

            if args.limit_frames and frame_count >= args.limit_frames:
                LOGGER.info("Reached frame limit (%d). Stopping.", args.limit_frames)
                break
        else:
            if LOGGER.isEnabledFor(logging.DEBUG):
                now = time.time()
                if now - last_log_time > 5:
                    LOGGER.debug("No new frames yet; still waiting...")

    finally:
        if writer is not None:
            writer.release()
        response.close()
        LOGGER.info("Recorded %d frames to %s", frame_count, args.output)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    def handle_interrupt(signum, frame):  # noqa: ARG001 - required signature
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    try:
        record_stream(args)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user. Output saved to %s", args.output)
    except Exception as exc:
        LOGGER.error("Recording failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
