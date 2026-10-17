#!/usr/bin/env bash
# Download the GGUF files for the quantisation ladder from the official Qwen repos
# (Apache-2.0). Retries on transient DNS/network errors and verifies each file's
# size against the server's Content-Length before accepting it.
#
#   bash scripts/download_models.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fetch() {  # fetch <repo> <file> <dest-dir>
  local repo="$1" file="$2" dir="$3"
  local url="https://huggingface.co/${repo}/resolve/main/${file}"
  mkdir -p "$dir"
  local dest="$dir/$file"
  local want
  for attempt in 1 2 3 4 5 6; do
    want=$(curl -sIL --max-time 30 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
    if [ -n "$want" ] && [ "$want" -gt 1000000 ]; then break; fi
    echo "  could not read size for $file (attempt $attempt); retrying"; sleep 10
  done
  if [ -z "$want" ] || [ "$want" -le 1000000 ]; then echo "FAILED (no size): $file"; return 1; fi
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then echo "ok (already complete): $file"; return 0; fi
  for attempt in 1 2 3 4 5 6; do
    echo "downloading $file ($want bytes), attempt $attempt"
    curl -sL --fail --retry 5 --retry-all-errors --retry-delay 5 -o "$dest" "$url" || { sleep 10; continue; }
    if [ "$(stat -c %s "$dest")" = "$want" ]; then echo "ok: $file"; return 0; fi
    echo "  size mismatch for $file; retrying"; sleep 5
  done
  echo "FAILED: $file"; return 1
}

status=0
R15=Qwen/Qwen2.5-1.5B-Instruct-GGUF
D15="$ROOT/models/qwen2.5-1.5b-instruct"
for f in qwen2.5-1.5b-instruct-q4_k_m.gguf qwen2.5-1.5b-instruct-q8_0.gguf qwen2.5-1.5b-instruct-fp16.gguf; do
  fetch "$R15" "$f" "$D15" || status=1
done

R7=Qwen/Qwen2.5-7B-Instruct-GGUF
D7="$ROOT/models/qwen2.5-7b-instruct"
for n in 1 2 3; do
  fetch "$R7" "qwen2.5-7b-instruct-q8_0-0000${n}-of-00003.gguf" "$D7" || status=1
done

echo "download script finished with status $status"
exit $status
