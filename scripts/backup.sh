#!/bin/bash
# Usage: ./scripts/backup.sh — creates a timestamped zip of edax_evals.txt and trained_weights.bin
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
name="backup_$(date +%Y-%m-%d_%H-%M).zip"
zip -j "$repo/ignored/$name" "$repo/ignored/edax_evals.txt" "$repo/ignored/trained_weights.bin"
echo "Created ignored/$name"
