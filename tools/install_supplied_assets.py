"""Copy the supplied meshes / velocity fields into the applications' output folders.

    python tools/install_supplied_assets.py            # install (verifies SHA-256 first)
    python tools/install_supplied_assets.py --check    # verify only, copy nothing

Reads ``supplied_assets_manifest.json`` at the repository root.  Each file under
``Applications/*/supplied/`` is checked against its recorded SHA-256 and copied
to ``install_to`` (inside the git-ignored ``output/`` folders, where the drivers
look for them).  An existing destination with the same hash is left alone; one
with a DIFFERENT hash is never overwritten (the script stops and reports it), so
a regenerated field is not silently replaced or mixed with a supplied one.

Standard library only; works in either Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "supplied_assets_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the supplied files only")
    args = ap.parse_args()

    entries = json.loads(MANIFEST.read_text())["files"]
    bad = []
    for e in entries:
        src = REPO / e["path"]
        if not src.is_file():
            bad.append(f"missing supplied file {e['path']}")
        elif sha256(src) != e["sha256"]:
            bad.append(f"sha256 mismatch for supplied file {e['path']}")
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    print(f"verified {len(entries)} supplied files")
    if args.check:
        return 0

    conflicts = []
    for e in entries:
        src, dst = REPO / e["path"], REPO / e["install_to"]
        if dst.exists():
            if sha256(dst) == e["sha256"]:
                print(f"  present  {e['install_to']}")
                continue
            conflicts.append(e["install_to"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied   {e['install_to']}")
    if conflicts:
        print("NOT overwritten (a different file is already there; it was regenerated or comes from "
              "another version -- move it away to use the supplied asset):", file=sys.stderr)
        print("\n".join(f"  {c}" for c in conflicts), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
