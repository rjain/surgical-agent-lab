#!/usr/bin/env python3
"""Cut the source video down to the short windows the lab actually uses.

**Instructor task, once.** Turns the multi-hour case videos fetched by
``tools/fetch_video.py`` into one small clip per flagged moment, and writes
``lab/clips.json`` so the repo knows where they are.

    python tools/cut_clips.py --dry-run     # what would be cut, and how much
    python tools/cut_clips.py               # cut them and write the manifest

Why the clips are small
-----------------------
A flagged segment is a whole task step: median eleven minutes, up to
forty-five. Sending one to the model costs roughly 170,000 tokens, which at a
shared rate limit is about six calls a minute for the entire project. Each
``Deviation`` therefore nominates a *watch window* — forty seconds around the
instant that explains the flag — and that is what gets cut here, with five
seconds of lead-in either side.

The clips are also re-encoded down from 1280x720 at 60 fps. The model samples
video at about one frame per second, so the original frame rate is bytes
nobody reads, and smaller clips upload faster in a room of twenty-five people.

ffmpeg comes from the ``imageio-ffmpeg`` package rather than the system, so
this works on a machine with nothing installed::

    pip install imageio-ffmpeg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.clips import CLIP_DIR, MANIFEST_PATH, clip_id_for  # noqa: E402
from lab.config import REPO_ROOT  # noqa: E402
from lab.rules import find_deviations  # noqa: E402
from tools.fetch_video import CURATED, VIDEO_DIR  # noqa: E402

#: Seconds of context either side of the watch window.
PAD_S = 5.0

#: Re-encode target. Wide enough to see instruments, small enough to upload.
WIDTH = 854
FPS = 15
CRF = 28


def ffmpeg() -> str:
    """Path to an ffmpeg binary, from the imageio-ffmpeg package."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "no ffmpeg available. Install it with:  pip install imageio-ffmpeg"
        ) from exc


def source_for(case_id: str, part: int) -> Path | None:
    """The downloaded video for one part of a case, if it is present."""
    candidate = VIDEO_DIR / f"{case_id}_video_part_{part:03d}.mp4"
    return candidate if candidate.is_file() else None


def cut(exe: str, source: Path, start_s: float, duration_s: float, target: Path) -> None:
    """Extract one window, re-encoded small.

    Seeks before the input for speed, then re-encodes so the cut is frame
    accurate — copying the stream would snap to the nearest keyframe and drift
    the window by seconds, which matters when the whole clip is forty of them.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            exe, "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{start_s:.3f}", "-i", str(source), "-t", f"{duration_s:.3f}",
            "-vf", f"scale={WIDTH}:-2,fps={FPS}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF),
            "-an", "-movflags", "+faststart",
            str(target),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case ids; defaults to the curated six")
    parser.add_argument("--dry-run", action="store_true", help="report without cutting")
    args = parser.parse_args()

    cases = args.cases or CURATED
    exe = None if args.dry_run else ffmpeg()

    manifest: dict[str, dict] = {}
    cut_count = missing = 0
    total_bytes = 0

    print(f"{'clip':30} {'window':>18} {'source':>10}  size")
    print("-" * 74)
    for case_id in cases:
        for dev in find_deviations(case_id):
            lo, hi = dev.watch_window
            start = max(0.0, lo - PAD_S)
            duration = (hi - lo) + 2 * PAD_S
            clip_id = clip_id_for(case_id, dev.part, start)
            target = CLIP_DIR / f"{clip_id}.mp4"

            source = source_for(case_id, dev.part)
            if source is None:
                missing += 1
                print(f"{clip_id:30} {f'{lo:.0f}-{hi:.0f}s':>18} {'MISSING':>10}  "
                      f"run fetch_video.py first")
                continue

            if args.dry_run:
                print(f"{clip_id:30} {f'{lo:.0f}-{hi:.0f}s':>18} {'ok':>10}  would cut")
            else:
                if not target.is_file():
                    cut(exe, source, start, duration, target)
                size = target.stat().st_size
                total_bytes += size
                cut_count += 1
                print(f"{clip_id:30} {f'{lo:.0f}-{hi:.0f}s':>18} {'ok':>10}  "
                      f"{size / 1e6:5.1f} MB")

            manifest[clip_id] = {
                "case_id": case_id,
                "part": dev.part,
                "start_s": round(start, 1),
                "end_s": round(start + duration, 1),
                "rule_id": dev.rule_id,
                "local": f"data/clips/{clip_id}.mp4",
                "uri": "",
            }

    if not args.dry_run and manifest:
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "note": (
                        "Generated by tools/cut_clips.py. Clips are cut to each "
                        "flag's watch window with 5s of padding, so start_s "
                        "matches the clip. Files API URIs live 48h; "
                        "resolve_clip re-uploads when they lapse."
                    ),
                    "clips": manifest,
                },
                indent=2,
            )
            + "\n"
        )

    print("-" * 74)
    print(f"{cut_count} clip(s), {total_bytes / 1e6:.0f} MB total"
          + (f", {missing} missing source video" if missing else ""))
    if not args.dry_run and manifest:
        print(f"manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)} ({len(manifest)} entries)")
        print("Next: python tools/upload_clips.py")
    return 1 if missing and not args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
