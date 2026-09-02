#!/usr/bin/env python3
"""Pre-upload the curated clips and record their URIs in lab/clips.json.

**Instructor task, on the morning of the session.** Files live for 48 hours,
so this cannot usefully be done more than two days ahead — and if it is not
done at all, every participant uploads their own copies and Lab 2 starts slower.

    python tools/upload_clips.py --check     # what the manifest says, and whether it is live
    python tools/upload_clips.py             # upload anything missing or expired

Run it with the project key, so the URIs it records are visible to every
participant using a key from that project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.clips import CLIP_DIR, MANIFEST_PATH, _upload, _uri_is_reachable, load_manifest  # noqa: E402


def read_raw() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"clips": {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report status without uploading"
    )
    args = parser.parse_args()

    manifest = load_manifest()
    if not manifest:
        print(f"No manifest at {MANIFEST_PATH}.")
        print("Generate one from the curated cases first, then re-run.")
        return 1

    raw = read_raw()
    print(f"{'clip':26} {'local':>7} {'uri':>10} {'action':>12}")
    print("-" * 60)
    uploaded = failed = live = pending = 0

    for clip_id, clip in sorted(manifest.items()):
        local = CLIP_DIR / f"{clip_id}.mp4"
        has_local = local.is_file() or (clip.local and Path(clip.local).is_file())
        reachable = bool(clip.uri) and _uri_is_reachable(clip.uri)

        if reachable:
            live += 1
            print(f"{clip_id:26} {'yes' if has_local else '-':>7} {'live':>10} {'-':>12}")
            continue
        if args.check:
            if has_local:
                pending += 1
            else:
                failed += 1
            print(f"{clip_id:26} {'yes' if has_local else '-':>7} "
                  f"{'stale' if clip.uri else 'none':>10} {'would upload' if has_local else 'NO SOURCE':>12}")
            continue
        if not has_local:
            failed += 1
            print(f"{clip_id:26} {'-':>7} {'none':>10} {'NO SOURCE':>12}")
            continue
        try:
            source = local if local.is_file() else Path(clip.local)
            uri = _upload(source)
            raw["clips"].setdefault(clip_id, {})["uri"] = uri
            uploaded += 1
            print(f"{clip_id:26} {'yes':>7} {'new':>10} {'uploaded':>12}")
        except Exception as exc:
            failed += 1
            print(f"{clip_id:26} {'yes':>7} {'-':>10} {type(exc).__name__:>12}")

    if not args.check and uploaded:
        MANIFEST_PATH.write_text(json.dumps(raw, indent=2) + "\n")
        print(f"\nwrote {MANIFEST_PATH}")

    done = f"{uploaded} uploaded" if not args.check else f"{pending} to upload"
    print(f"\n{live} already live, {done}, {failed} without a source")
    if failed:
        print(f"Put the missing clips in {CLIP_DIR} and re-run.")
    if args.check and pending:
        print("Re-run without --check to upload them.")
    # --check reports; it is not a failure to have work outstanding.
    return 1 if failed and not args.check else 0


if __name__ == "__main__":
    sys.exit(main())
