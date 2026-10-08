"""Safely remove generated Test Application3 optimization outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (APP_DIR / "output").resolve()
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
            f"Refusing to delete target outside App3 output: {resolved}"
        ) from exc
    if resolved == OUTPUT_ROOT:
        raise ValueError("Refusing to delete the App3 output directory itself")
    return resolved


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
    safe_target = require_inside_output(target)
    files, folders = count_entries(safe_target)
    action = "Would delete" if dry_run else "Deleting"
    print(f"{action}: {safe_target}")

    if dry_run:
        return files, folders
    if safe_target.is_symlink() or safe_target.is_file():
        safe_target.unlink()
    elif safe_target.is_dir():
        shutil.rmtree(safe_target)
    return files, folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely remove generated Test Application3 outputs."
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
        "--dry-run",
        action="store_true",
        help="Print deletion targets without deleting them.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of all App3 output contents without prompting.",
    )
    args = parser.parse_args()
    if args.mode is not None and args.case_name is None:
        parser.error("--mode requires --case-name")
    return args


def confirm_full_cleanup() -> bool:
    prompt = (
        f"Delete all contents inside {OUTPUT_ROOT} while preserving the "
        "output directory? [y/N]: "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.case_name is not None:
        case_name = validate_case_name(args.case_name)
        target = OUTPUT_ROOT / case_name
        if args.mode is not None:
            target = target / args.mode
        require_inside_output(target)
        targets = [target] if target.exists() or target.is_symlink() else []
        target_description = str(target.resolve())
    else:
        target_description = f"contents of {OUTPUT_ROOT}"
        targets = list(OUTPUT_ROOT.iterdir())
        for target in targets:
            require_inside_output(target)
        if not args.dry_run and not args.yes and not confirm_full_cleanup():
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
    print(
        f"Summary: {files_deleted} files and {folders_deleted} folders "
        f"{verb}."
    )
    if args.dry_run:
        print("Dry run complete; no files were deleted.")
    else:
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
