#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


from .disord_utils import (
    normalise_cif_file,
    process_cif_atom_line,
)


process_cif_line = process_cif_atom_line

def process_file(
    path: str | Path,
    make_backup: bool = True,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Normalise one CIF file."""
    return normalise_cif_file(
        path,
        make_backup=make_backup,
        dry_run=dry_run,
    )


def walk_and_process(
    root: str | Path,
    recursive: bool = True,
    exts: list[str] | None = None,
    make_backup: bool = True,
    dry_run: bool = False,
) -> dict[str, list]:
    """Normalise matching files below ``root`` and collect their statuses."""
    root = Path(root)
    results = {"modified": [], "unchanged": [], "skipped": [], "errors": []}
    extensions = {extension.lower() for extension in exts or []}

    if recursive:
        walker = os.walk(root)
    else:
        walker = [(str(root), [], os.listdir(root))]

    for directory, _, filenames in walker:
        for filename in filenames:
            path = Path(directory) / filename
            if extensions and path.suffix.lower() not in extensions:
                results["skipped"].append((str(path), "ext_mismatch"))
                continue

            print(f"Scanning: {filename}...", end="\r")
            ok, status = process_file(
                path,
                make_backup=make_backup,
                dry_run=dry_run,
            )
            if not ok:
                results["errors"].append((str(path), status))
            elif status == "modified":
                results["modified"].append(str(path))
            elif status == "unchanged":
                results["unchanged"].append(str(path))
            elif status == "would_modify":
                results["modified"].append(f"{path} (dry-run)")
            else:
                results["skipped"].append((str(path), status))

    print(" " * 50, end="\r")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean CIFs: round coordinates and renumber atom labels."
    )
    parser.add_argument("root", help="Directory to process")
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".cif"],
        help="Extensions to process (default: .cif)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_false",
        dest="backup",
        help="Do not create .bak files",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    if not Path(args.root).is_dir():
        print(f"Error: {args.root} is not a directory.")
        return

    results = walk_and_process(
        args.root,
        recursive=args.recursive,
        exts=args.ext,
        make_backup=args.backup,
        dry_run=args.dry_run,
    )
    print("\nSummary:")
    print(f"  Modified: {len(results['modified'])}")
    print(f"  Unchanged: {len(results['unchanged'])}")
    print(f"  Errors: {len(results['errors'])}")
    for filename, error in results["errors"]:
        print(f"    {filename} -> {error}")


if __name__ == "__main__":
    main()
