#!/usr/bin/env python3
"""Download Othello/Reversi PGN games from playok.com.

URL pattern: https://www.playok.com/p/?g={prefix}{id}.txt
IDs are sequential integers. Files are created in aligned 1000-game chunks
with names like playok_pgn_75268000.pgn (start ID always divisible by 1000).

Usage:
  python3 scripts/playok_download.py --start 75268000 --end 75269999
  python3 scripts/playok_download.py --start 75268000 --end 75269999 --prefix rv
"""

import argparse
import os
import sys
import time

import requests

BASE_URL = "https://www.playok.com/p/"
DEFAULT_PREFIX = "rv"
CHUNK_SIZE = 1000
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "ignored")
REQUEST_TIMEOUT = 30  # seconds
RETRY_DELAY = 2  # seconds between retries
MAX_RETRIES = 3


def download_game(session: requests.Session, game_id: int, prefix: str) -> str | None:
    """Download a single game, returning its text or None on failure."""
    url = f"{BASE_URL}?g={prefix}{game_id}.txt"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                text = resp.text.strip()
                if text:
                    return text
                return None  # empty response, likely invalid ID
            elif resp.status_code == 404:
                return None  # game doesn't exist
            else:
                print(f"  HTTP {resp.status_code} for {prefix}{game_id} (attempt {attempt})")
        except requests.RequestException as e:
            print(f"  Request error for {prefix}{game_id}: {e} (attempt {attempt})")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
    return None


def ensure_output_dir() -> None:
    """Create the output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_alignment(start: int, end: int) -> None:
    """Ensure start and end+1 are both divisible by CHUNK_SIZE."""
    if start % CHUNK_SIZE != 0:
        print(f"Error: --start ({start}) must be divisible by {CHUNK_SIZE}")
        print(f"  e.g. --start {(start // CHUNK_SIZE) * CHUNK_SIZE} or "
              f"--start {((start // CHUNK_SIZE) + 1) * CHUNK_SIZE}")
        sys.exit(1)
    if (end + 1) % CHUNK_SIZE != 0:
        print(f"Error: --end+1 ({end + 1}) must be divisible by {CHUNK_SIZE}")
        print(f"  e.g. --end {((end + 1) // CHUNK_SIZE) * CHUNK_SIZE - 1} or "
              f"--end {((end + 1) // CHUNK_SIZE + 1) * CHUNK_SIZE - 1}")
        sys.exit(1)


def download_chunk(
    session: requests.Session,
    chunk_start: int,
    chunk_end: int,
    prefix: str,
) -> tuple[int, int]:
    """Download all games in [chunk_start, chunk_end] to a single file.

    Returns (downloaded, total).
    """
    filename = f"playok_pgn_{chunk_start}.pgn"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"  {filename} already exists, skipping")
        total = chunk_end - chunk_start + 1
        return 0, total

    downloaded = 0
    total = chunk_end - chunk_start + 1
    print(f"  Downloading {total} games → {filename} ...")

    with open(filepath, "w") as f:
        for game_id in range(chunk_start, chunk_end + 1):
            text = download_game(session, game_id, prefix)
            if text:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
                # Ensure blank line between games (PGN convention)
                if not text.endswith("\n\n"):
                    f.write("\n")
                downloaded += 1

            # Progress every 100 games
            if (game_id - chunk_start + 1) % 100 == 0:
                pct = (game_id - chunk_start + 1) * 100 // total
                print(f"    [{pct:3d}%] {game_id - chunk_start + 1}/{total} ({downloaded} found)")

    if downloaded == 0:
        os.remove(filepath)
        print(f"    No games found in range, removed {filename}")
    else:
        print(f"    Saved {downloaded}/{total} games to {filename}")

    return downloaded, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Othello PGN games from playok.com"
    )
    parser.add_argument(
        "--start", type=int, required=True,
        help="First game ID to download (e.g. 75268000)"
    )
    parser.add_argument(
        "--end", type=int, required=True,
        help="Last game ID to download (inclusive, e.g. 75269999)"
    )
    parser.add_argument(
        "--prefix", type=str, default=DEFAULT_PREFIX,
        help=f"Game ID prefix (default: {DEFAULT_PREFIX})"
    )
    args = parser.parse_args()

    if args.start > args.end:
        print("Error: --start must be <= --end")
        sys.exit(1)

    validate_alignment(args.start, args.end)
    ensure_output_dir()

    n_files = (args.end - args.start + 1) // CHUNK_SIZE
    print(f"Range: {args.prefix}{args.start} → {args.prefix}{args.end}")
    print(f"Total IDs: {args.end - args.start + 1} across {n_files} file(s)")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "playok-downloader/1.0 (othello training data collector)"
    })

    total_downloaded = 0
    total_attempted = 0

    for chunk_start in range(args.start, args.end + 1, CHUNK_SIZE):
        chunk_end = chunk_start + CHUNK_SIZE - 1
        downloaded, attempted = download_chunk(session, chunk_start, chunk_end, args.prefix)
        total_downloaded += downloaded
        total_attempted += attempted

    print()
    print(f"Done: {total_downloaded}/{total_attempted} games downloaded")


if __name__ == "__main__":
    main()
