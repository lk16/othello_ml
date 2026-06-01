#!/usr/bin/env python3
"""Download Othello/Reversi PGN games from playok.com.

URL pattern: https://www.playok.com/p/?g=rv{id}.txt
IDs are sequential integers. Files are created in aligned 1000-game chunks
with names like playok_pgn_75268000.pgn (start ID always divisible by 1000).

Usage:
  python3 scripts/playok_download.py --start 75268000 --end 75269999
  python3 scripts/playok_download.py --start 75268000 --end 75269999 --threads 8
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = "https://www.playok.com/p/"
CHUNK_SIZE = 1000
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "training_data")
REQUEST_TIMEOUT = 30  # seconds
RETRY_DELAY = 2  # seconds between retries
MAX_RETRIES = 3

_progress_lock = threading.Lock()


def download_game(game_id: int) -> tuple[int, str | None]:
    """Download a single game, returning (game_id, text_or_None)."""
    url = f"{BASE_URL}?g=rv{game_id}.txt"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                text = resp.text.strip()
                return (game_id, text if text else None)
            elif resp.status_code == 404:
                return (game_id, None)
        except requests.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return (game_id, None)


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
    chunk_start: int,
    chunk_end: int,
    threads: int,
) -> tuple[int, int]:
    """Download all games in [chunk_start, chunk_end] in parallel, write to file.

    Returns (downloaded, total).
    """
    filename = f"playok_pgn_{chunk_start}.pgn"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        print(f"  {filename} already exists, skipping")
        return 0, chunk_end - chunk_start + 1

    total = chunk_end - chunk_start + 1
    game_ids = list(range(chunk_start, chunk_end + 1))
    print(f"  Downloading {total} games → {filename} ({threads} thread(s)) ...")

    results: dict[int, str | None] = {}
    downloaded = 0
    done = 0

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(download_game, gid): gid for gid in game_ids}

        for future in as_completed(futures):
            gid, text = future.result()
            results[gid] = text
            if text:
                downloaded += 1
            done += 1

            # Progress every 100 games (thread-safe)
            if done % 100 == 0 or done == total:
                pct = done * 100 // total
                with _progress_lock:
                    print(f"    [{pct:3d}%] {done}/{total} ({downloaded} found)")

    if downloaded == 0:
        print(f"    No games found in range, skipping {filename}")
        return 0, total

    # Write in order
    with open(filepath, "w") as f:
        for gid in game_ids:
            text = results.get(gid)
            if text:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
                if not text.endswith("\n\n"):
                    f.write("\n")

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
        "--threads", type=int, default=1,
        help="Number of parallel download threads (default: 1)"
    )
    args = parser.parse_args()

    if args.start > args.end:
        print("Error: --start must be <= --end")
        sys.exit(1)
    if args.threads < 1:
        print("Error: --threads must be >= 1")
        sys.exit(1)

    validate_alignment(args.start, args.end)
    ensure_output_dir()

    n_files = (args.end - args.start + 1) // CHUNK_SIZE
    print(f"Range: rv{args.start} → rv{args.end}")
    print(f"Total IDs: {args.end - args.start + 1} across {n_files} file(s)")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    total_downloaded = 0
    total_attempted = 0

    for chunk_start in range(args.start, args.end + 1, CHUNK_SIZE):
        chunk_end = chunk_start + CHUNK_SIZE - 1
        downloaded, attempted = download_chunk(chunk_start, chunk_end, args.threads)
        total_downloaded += downloaded
        total_attempted += attempted

    print()
    print(f"Done: {total_downloaded}/{total_attempted} games downloaded")


if __name__ == "__main__":
    main()
