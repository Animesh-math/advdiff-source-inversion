"""Safely remove generated Test Application4 outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


APP_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = (APP_DIR / "output").resolve()
MESH_ROOT = (APP_DIR / "mesh").resolve()
MODE_FOLDERS = ("amplitude_only", "location_only", "full")
CASE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
GENERATED_MESH_NAMES = (
    "l_shaped_city_with_buildings.msh",
    "l_shaped_city_with_buildings.vtk",
)


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
            f"Refusing to delete target outside App4 output: {resolved}"
        ) from exc
    if resolved == OUTPUT_ROOT:
        raise ValueError("Refusing to delete the App4 output directory itself")
    return resolved


def require_generated_mesh_file(target: Path) -> Path:
    resolved = target.resolve()
    if resolved.parent != MESH_ROOT or resolved.name not in GENERATED_MESH_NAMES:
        raise ValueError(f"Refusing to delete non-App4 mesh artifact: {resolved}")
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
        description="Safely remove generated Test Application4 outputs."
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
        "--mesh",
        action="store_true",
        help="Also delete generated App4 mesh .msh/.vtk artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deletion targets without deleting them.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of all App4 output contents without prompting.",
    )
    args = parser.parse_args()
    if args.mode is not None and args.case_name is None:
        parser.error("--mode requires --case-name")
    return args


def confirm_full_cleanup(include_mesh: bool) -> bool:
    mesh_text = " and generated mesh artifacts" if include_mesh else ""
    prompt = (
        f"Delete all contents inside {OUTPUT_ROOT}{mesh_text} while preserving "
        "the output and mesh directories? [y/N]: "
    )
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MESH_ROOT.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.case_name is not None:
        case_name = validate_case_name(args.case_name)
        target = OUTPUT_ROOT / case_name
        if args.mode is not None:
            target = target / args.mode
        require_inside_output(target)
        targets.extend([target] if target.exists() or target.is_symlink() else [])
        target_description = str(target.resolve())
    else:
        target_description = f"contents of {OUTPUT_ROOT}"
        targets.extend(OUTPUT_ROOT.iterdir())
        for target in targets:
            require_inside_output(target)
        if not args.dry_run and not args.yes and not confirm_full_cleanup(args.mesh):
            print("Cleanup cancelled; nothing was deleted.")
            return

    if args.mesh:
        for mesh_name in GENERATED_MESH_NAMES:
            mesh_target = MESH_ROOT / mesh_name
            require_generated_mesh_file(mesh_target)
            if mesh_target.exists() or mesh_target.is_symlink():
                targets.append(mesh_target)
        target_description += f"; generated mesh artifacts in {MESH_ROOT}"

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
