#!/usr/bin/env python3
"""Download the SurgVU labels. Run this once, during setup.

    python tools/fetch_labels.py

This is the one tool participants do run. It pulls the published label
archive — 335 KB, all 155 cases — and unpacks it to ``data/labels/`` where the
lab looks for it by default. Takes a couple of seconds.

Two things worth knowing:

* **You are downloading from the source, not from us.** The archive is the
  published SurgVU release on its own public bucket, so the dataset's own
  terms apply to you directly and nothing has been repackaged in between.
* **These are labels, not video.** Task segments and instrument mounts as CSV.
  The video clips the lab uses are handled separately by ``lab/clips.py``; the
  full video archive is 344 GB and nobody needs it.

Pass ``--force`` to re-download over an existing copy.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.config import REPO_ROOT  # noqa: E402

#: The published label release. Verified byte-identical to the copy the lab
#: was built and measured against.
ARCHIVE = "https://storage.googleapis.com/isi-surgvu/surgvu24_labels_updated_v2.zip"

TARGET = REPO_ROOT / "data" / "labels"

#: Jupyter leaves these all over the archive; they are not data.
NOISE = ".ipynb_checkpoints"


def already_there() -> int:
    """How many cases are unpacked at the target already."""
    if not TARGET.is_dir():
        return 0
    return sum(1 for d in TARGET.iterdir() if d.is_dir() and d.name.startswith("case_"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download and replace")
    args = parser.parse_args()

    if TARGET.is_symlink():
        print(f"{TARGET} is a symlink to {TARGET.resolve()} — leaving it alone.")
        return 0

    have = already_there()
    if have and not args.force:
        print(f"{have} cases already at {TARGET.relative_to(REPO_ROOT)}. "
              "Nothing to do (--force to replace).")
        return 0

    print(f"downloading {ARCHIVE.rsplit('/', 1)[-1]} …", flush=True)
    with urllib.request.urlopen(ARCHIVE, timeout=120) as response:
        blob = response.read()
    print(f"  {len(blob) / 1e3:.0f} KB")

    TARGET.mkdir(parents=True, exist_ok=True)
    cases: set[str] = set()
    written = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            if info.is_dir() or NOISE in info.filename:
                continue
            if not info.filename.lower().endswith(".csv"):
                continue
            # The archive nests everything under `labels/`; drop that so the
            # result is data/labels/case_045/tasks.csv, which is what the code
            # looks for with nothing configured.
            parts = Path(info.filename).parts
            if "labels" in parts:
                parts = parts[parts.index("labels") + 1:]
            if len(parts) != 2:
                continue
            case, name = parts
            cases.add(case)
            destination = TARGET / case / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            written += 1

    print(f"{len(cases)} cases, {written} CSV files -> "
          f"{TARGET.relative_to(REPO_ROOT)}")
    if len(cases) != 155:
        print(f"  warning: expected 155 cases, got {len(cases)}")
        return 1
    print("Next: python preflight.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
