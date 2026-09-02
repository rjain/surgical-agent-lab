#!/usr/bin/env python3
"""Pull only the cases we need out of the 344 GB SurgVU video archive.

**Instructor task, once.** Participants never run this — they get 40-second
clips, not source video.

The published archive is a single 344 GB zip. It is also a *public* object on
Cloud Storage, which honours HTTP range requests, and it stores one video per
case. So there is no need to download all of it: read the zip's central
directory over ranges, find the members for the cases we curated, and stream
just those. The six curated cases come to about 11.6 GB.

    python tools/fetch_video.py --list                 # what is in the archive
    python tools/fetch_video.py --curated              # fetch the six we use
    python tools/fetch_video.py case_045 case_036      # or name your own

Downloads land in ``data/video/`` and are skipped if already complete, so an
interrupted run can simply be repeated. Next step is ``tools/cut_clips.py``,
which cuts these into the short windows the lab actually ships.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.config import REPO_ROOT  # noqa: E402

ARCHIVE = "https://storage.googleapis.com/isi-surgvu/surgvu24_videos_only.zip"
VIDEO_DIR = REPO_ROOT / "data" / "video"

#: The cases chosen on measured rule coverage; see tools/evaluate_rules.py.
CURATED = ["case_045", "case_129", "case_125", "case_036", "case_044", "case_059"]


class RangeFile(io.RawIOBase):
    """A seekable read-only file over an HTTP object, using Range requests.

    Enough for :mod:`zipfile` to read a central directory without pulling the
    whole archive.

    Reads retry on their own. An eleven-gigabyte pull over a hotel or office
    link will drop a connection sooner or later, and because each read is an
    independent ranged request, retrying one costs a few megabytes rather than
    restarting the member. Without this a single timeout three hours in
    discards everything.
    """

    def __init__(self, url: str, timeout: int = 180, attempts: int = 6) -> None:
        self.url = url
        self.timeout = timeout
        self.attempts = attempts
        self.pos = 0
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=timeout) as response:
            self.size = int(response.headers["Content-Length"])

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.pos
        if size <= 0:
            return b""
        last = min(self.pos + size - 1, self.size - 1)
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{last}"}
        )
        for attempt in range(1, self.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == self.attempts:
                    raise
                # The position has not moved, so the same range is simply
                # asked for again. Back off a little each time.
                delay = min(2 ** attempt, 30)
                print(
                    f"    read failed ({type(exc).__name__}), retrying in "
                    f"{delay}s [{attempt}/{self.attempts - 1}]",
                    flush=True,
                )
                time.sleep(delay)
        self.pos += len(data)
        return data


def members_for(archive: zipfile.ZipFile, cases: list[str]) -> list[zipfile.ZipInfo]:
    """Video members belonging to the named cases, smallest first."""
    wanted = []
    for info in archive.infolist():
        if info.is_dir() or "__MACOSX" in info.filename:
            continue
        if not info.filename.lower().endswith(".mp4"):
            continue
        if any(f"/{case}/" in info.filename for case in cases):
            wanted.append(info)
    return sorted(wanted, key=lambda i: i.file_size)


def fetch(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Path:
    """Stream one member to ``data/video/``, skipping it if already complete."""
    target = VIDEO_DIR / Path(info.filename).name
    if target.is_file() and target.stat().st_size == info.file_size:
        print(f"  {target.name:44} {info.file_size / 1e9:5.2f} GB  already have it")
        return target

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"  {target.name:44} {info.file_size / 1e9:5.2f} GB  downloading…", flush=True)
    with archive.open(info) as source, partial.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=8 << 20)
    partial.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case ids, e.g. case_045")
    parser.add_argument("--curated", action="store_true", help="the six we use")
    parser.add_argument("--list", action="store_true", help="list the archive and exit")
    args = parser.parse_args()

    print(f"reading the archive directory over range requests…", flush=True)
    archive = zipfile.ZipFile(RangeFile(ARCHIVE))

    if args.list:
        cases: dict[str, int] = {}
        for info in archive.infolist():
            if info.filename.lower().endswith(".mp4") and "__MACOSX" not in info.filename:
                case = Path(info.filename).parent.name
                cases[case] = cases.get(case, 0) + info.file_size
        print(f"{len(cases)} cases in the archive")
        for case, size in sorted(cases.items()):
            mark = " *" if case in CURATED else ""
            print(f"  {case:12} {size / 1e9:6.2f} GB{mark}")
        print("\n* = curated for this lab")
        return 0

    cases = CURATED if args.curated else args.cases
    if not cases:
        parser.error("name some cases, or pass --curated")

    wanted = members_for(archive, cases)
    if not wanted:
        print(f"no video members matched {cases}")
        return 1

    total = sum(i.file_size for i in wanted)
    print(f"{len(wanted)} member(s), {total / 1e9:.1f} GB total\n")
    for info in wanted:
        fetch(archive, info)
    print(f"\nin {VIDEO_DIR}. Next: python tools/cut_clips.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
