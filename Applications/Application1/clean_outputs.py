"""Safely remove generated Application1 optimization/diagnostic outputs.

Adapted from ``Test Application4/clean_outputs.py`` for the DC Navy Yard app.

CRITICAL SAFETY DIFFERENCE FROM APP4:
The expensive Application1 Navier-Stokes result lives under
``output/navier_stokes/`` (the nodal velocity NPZ that the whole optimization
pipeline consumes).  The ordinary cleanup NEVER touches it.  It is only removed
when the explicit, loudly-warned ``--include-navier-stokes`` flag is given (a
full reset).  There is no generated-mesh option: the DC mesh is committed under
``DC_mesh/`` and is never a cleanup target here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (APP_DIR / "output").resolve()
NAVIER_STOKES_DIR = (OUTPUT_ROOT / "navier_stokes").resolve()
MODE_FOLDERS = ("amplitude_only", "location_only", "full")
CASE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def validate_case_name(case_name: str) -> str:
    if not CASE_NAME_PATTERN.fullmatch(case_name):
        raise ValueError(
            "case-name must start with an alphanumeric character and contain "
            "only letters, numbers, underscores, and hyphens"
        )
    return case_name


def require_inside_output(target: Path) -> Path:
    resolved = target.resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to delete target outside Application1 output: {resolved}"
        ) from exc
    if resolved == OUTPUT_ROOT:
        raise ValueError("Refusing to delete the Application1 output directory itself")
    return resolved


def is_navier_stokes(target: Path) -> bool:
    resolved = target.resolve()
    return resolved == NAVIER_STOKES_DIR or NAVIER_STOKES_DIR in resolved.parents


def count_entries(target: Path) -> tuple[int, int]:
    if target.is_symlink() or target.is_file():
        return (1, 0)
    if not target.is_dir():
        return (0, 0)

    files = 0
    folders = 1
    for child in target.iterdir():
        child_files, child_folders = count_entries(child)
        files += child_files
        folders += child_folders
    return files, folders


def remove_target(target: Path, dry_run: bool) -> tuple[int, int]:
    files, folders = count_entries(target)
    action = "Would delete" if dry_run else "Deleting"
    print(f"{action}: {target}")

    if dry_run:
        return files, folders
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    return files, folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely remove generated Application1 outputs (never the "
        "Navier-Stokes velocity unless --include-navier-stokes is given)."
    )
    parser.add_argument(
        "--case-name",
        help="Delete only output/<case-name>/.",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_FOLDERS,
        help="Delete only output/<case-name>/<mode>/; requires --case-name.",
    )
    parser.add_argument(
        "--include-navier-stokes",
        action="store_true",
        help=(
            "DANGER: also delete output/navier_stokes/ (the expensive velocity "
            "field). Only valid for a full reset (no --case-name)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deletion targets without deleting them.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion without prompting.",
    )
    args = parser.parse_args()
    if args.mode is not None and args.case_name is None:
        parser.error("--mode requires --case-name")
    if args.include_navier_stokes and args.case_name is not None:
        parser.error("--include-navier-stokes is only valid for a full reset")
    return args


def confirm_full_cleanup(include_navier_stokes: bool) -> bool:
    if include_navier_stokes:
        prompt = (
            f"DANGER: delete ALL of {OUTPUT_ROOT} INCLUDING the Navier-Stokes "
            "velocity field in output/navier_stokes/ (expensive to regenerate)? "
            "[y/N]: "
        )
    else:
        prompt = (
            f"Delete generated contents inside {OUTPUT_ROOT} while PRESERVING "
            "output/navier_stokes/ and the output directory? [y/N]: "
        )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    if args.case_name is not None:
        case_name = validate_case_name(args.case_name)
        target = OUTPUT_ROOT / case_name
        if args.mode is not None:
            target = target / args.mode
        require_inside_output(target)
        if is_navier_stokes(target):
            raise ValueError("Refusing to delete the Navier-Stokes output via --case-name")
        targets.extend([target] if target.exists() or target.is_symlink() else [])
        target_description = str(target.resolve())
    else:
        target_description = f"generated contents of {OUTPUT_ROOT}"
        for child in OUTPUT_ROOT.iterdir():
            require_inside_output(child)
            if is_navier_stokes(child) and not args.include_navier_stokes:
                print(f"Preserving Navier-Stokes velocity output: {child}")
                continue
            targets.append(child)
        if args.include_navier_stokes:
            target_description += " (INCLUDING output/navier_stokes/)"
        else:
            target_description += " (preserving output/navier_stokes/)"
        if not args.dry_run and not args.yes and not confirm_full_cleanup(
            args.include_navier_stokes
        ):
            print("Cleanup cancelled; nothing was deleted.")
            return

    print(f"Deletion target: {target_description}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")

    if not targets:
        print("Nothing to delete.")
        return

    files_deleted = 0
    folders_deleted = 0
    for target in targets:
        files, folders = remove_target(target, args.dry_run)
        files_deleted += files
        folders_deleted += folders

    verb = "would be deleted" if args.dry_run else "deleted"
    print(f"Summary: {files_deleted} files and {folders_deleted} folders {verb}.")
    if args.dry_run:
        print("Dry run complete; no files were deleted.")
    else:
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
