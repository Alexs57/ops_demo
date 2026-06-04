"""Download CWRU bearing dataset .mat files for OPS-Bench.

Two sources, tried in order per file:
  1. yyxyz/CaseWesternReserveUniversityData GitHub mirror (China-friendly).
     Optionally prefixed with ghproxy.com (use --use-ghproxy from China if raw github is slow).
  2. Official CWRU at engineering.case.edu/sites/default/files/<N>.mat (US-hosted, often slow from CN).

Files saved as ``data/cwru/<N>.mat`` to match what src.data.cwru.load_cwru expects.
Already-present, validated files are skipped (use --force to re-download).

Stdlib only — no extra dependencies beyond Python 3.

Examples:
  # Default (try yyxyz raw GitHub, then CWRU official)
  python scripts/download_cwru.py

  # From China with slow GitHub raw access
  python scripts/download_cwru.py --use-ghproxy

  # Only the 4 baseline files
  python scripts/download_cwru.py --files 97 98 99 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.cwru import DEFAULT_CLASS_MAP  # noqa: E402

YYXYZ_API = "https://api.github.com/repos/yyxyz/CaseWesternReserveUniversityData/contents"
YYXYZ_RAW = "https://raw.githubusercontent.com/yyxyz/CaseWesternReserveUniversityData/master"
GHPROXY = "https://ghproxy.com/"
CWRU_OFFICIAL = "https://engineering.case.edu/sites/default/files"

CHUNK_SIZE = 64 * 1024
UA = "ops-bench-downloader/1.0 (+https://github.com/Alexs57/ops_bench)"


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _http_download(url: str, dest: Path, timeout: int = 120) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    return total


def validate_mat(path: Path) -> bool:
    """Accept MATLAB v5 ('MATLAB' prefix) or v7.3 HDF5 (0x89 H D F)."""
    if not path.exists() or path.stat().st_size < 1024:
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    return head[:6] == b"MATLAB" or head[:4] == b"\x89HDF"


def get_yyxyz_filename_map(use_ghproxy: bool, timeout: int = 30) -> Dict[str, str]:
    """Map CWRU file numbers to actual filenames in the yyxyz mirror.

    The mirror uses descriptive names like ``12k_Drive_End_IR007_0_105.mat``.
    We list the repo root via GitHub Contents API, extract the trailing
    underscore-separated numeric token as the CWRU file number.
    Returns {} on API failure; downloader will fall back to the official site.
    """
    api = (GHPROXY + YYXYZ_API) if use_ghproxy else YYXYZ_API
    try:
        data = json.loads(_http_get(api, timeout=timeout))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as e:
        print(f"  [warn] yyxyz Contents API failed ({e!r}); will rely on official CWRU only.")
        return {}
    mapping: Dict[str, str] = {}
    for item in data:
        if item.get("type") != "file":
            continue
        name = item.get("name", "")
        if not name.endswith(".mat"):
            continue
        for token in reversed(name[:-4].split("_")):
            if token.isdigit():
                mapping[token] = name
                break
    return mapping


def download_one(
    file_num: str,
    dest: Path,
    yyxyz_filename: Optional[str],
    use_ghproxy: bool,
    timeout: int,
) -> Tuple[bool, str]:
    """Try yyxyz then official CWRU. Return (success, source_used_or_error)."""
    sources: List[Tuple[str, str]] = []
    if yyxyz_filename:
        url = f"{YYXYZ_RAW}/{urllib.parse.quote(yyxyz_filename)}"
        if use_ghproxy:
            url = GHPROXY + url
        sources.append(("yyxyz", url))
    sources.append(("cwru-official", f"{CWRU_OFFICIAL}/{file_num}.mat"))

    for source, url in sources:
        try:
            n_bytes = _http_download(url, dest, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"\n    [{source}] failed: {e!r}", end="")
            dest.unlink(missing_ok=True)
            continue
        if validate_mat(dest):
            return True, f"{source} ({n_bytes // 1024} KB)"
        print(f"\n    [{source}] downloaded {n_bytes} B but not a valid .mat, trying next", end="")
        dest.unlink(missing_ok=True)
    return False, "all sources failed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/cwru")
    p.add_argument("--use-ghproxy", action="store_true",
                   help="Prefix GitHub URLs with ghproxy.com (use from China if raw GitHub is slow).")
    p.add_argument("--files", nargs="+",
                   help="Specific file numbers to download (default: all in DEFAULT_CLASS_MAP).")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a valid file is already present.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    file_nums = args.files if args.files else sorted(set(DEFAULT_CLASS_MAP.keys()), key=int)
    print(f"Target: {len(file_nums)} files -> {data_root}/")
    if args.use_ghproxy:
        print("  using ghproxy.com prefix for GitHub URLs")

    print("Resolving yyxyz mirror filenames via GitHub Contents API...")
    yyxyz_map = get_yyxyz_filename_map(args.use_ghproxy, timeout=args.timeout)
    print(f"  got {len(yyxyz_map)} numeric->filename entries from yyxyz")

    successes: List[str] = []
    failures: List[str] = []
    for i, fn in enumerate(file_nums, 1):
        dest = data_root / f"{fn}.mat"
        if dest.exists() and not args.force and validate_mat(dest):
            print(f"[{i:>2}/{len(file_nums)}] {fn}.mat  skip (already valid)")
            successes.append(fn)
            continue
        print(f"[{i:>2}/{len(file_nums)}] {fn}.mat  ", end="", flush=True)
        ok, msg = download_one(fn, dest, yyxyz_map.get(fn), args.use_ghproxy, args.timeout)
        if ok:
            print(f"OK [{msg}]")
            successes.append(fn)
        else:
            print(f"\n    FAIL [{msg}]")
            failures.append(fn)
        time.sleep(0.1)  # gentle pacing

    print()
    print("=" * 60)
    print(f"Done. {len(successes)}/{len(file_nums)} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed:", " ".join(failures))
        print("Try re-running with --use-ghproxy, or grab the rest manually from:")
        print("  https://engineering.case.edu/bearingdatacenter")
        sys.exit(1)
    print("All files downloaded and validated. Next:")
    print(f"  python scripts/day1_pilot.py --dataset cwru --data-root {data_root} --epochs 30")


if __name__ == "__main__":
    main()
