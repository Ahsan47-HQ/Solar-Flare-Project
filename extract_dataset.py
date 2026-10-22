"""Extract the seven SWAN-SF archives into one shared Windows-safe layout.

Run from the repository root: python extract_dataset.py
The manifest retains original tar member names, labels, regions, and time ranges.
The command is resumable: existing files with the expected size are skipped.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVES = [ROOT / "SWAN.tar.gz", ROOT / "addenda.tar.gz"] + [
    ROOT / f"partition{i}_instances.tar.gz" for i in range(1, 6)
]
FIELDS = [
    "archive", "member", "path", "kind", "partition", "label",
    "region_id", "start", "end", "bytes",
]
WINDOW_RE = re.compile(r"_ar(?P<region>\d+)_s(?P<start>[^_]+)_e(?P<end>[^.]+)\.csv$")
MIN_FREE_BYTES = 750_000_000


def destination(archive: Path, member: tarfile.TarInfo) -> tuple[Path, str]:
    parts = PurePosixPath(member.name).parts
    parts = tuple(part for part in parts if part != ".")
    if not parts or any(part in ("..", "") for part in parts):
        raise ValueError(f"Unsafe archive path: {member.name}")
    if archive.name == "addenda.tar.gz":
        parts = ("addenda",) + parts
    else:
        expected = "SWAN" if archive.name == "SWAN.tar.gz" else archive.name.split("_instances")[0]
        if parts[0] != expected:
            raise ValueError(f"Unexpected archive root: {member.name}")
    # Percent encoding is reversible and avoids Windows' forbidden ':' in timestamps.
    safe_parts = [quote(part, safe="-_.@+") for part in parts]
    relative = Path(*safe_parts)
    return DATA / relative, relative.as_posix()


def manifest_row(archive: Path, member: tarfile.TarInfo, relative: str) -> dict[str, str | int]:
    name = member.name
    if archive.name == "SWAN.tar.gz":
        kind = "series"
        partition = PurePosixPath(name).parts[-2]
        region_id = PurePosixPath(name).stem
        label = start = end = ""
    elif archive.name == "addenda.tar.gz":
        kind = "addenda"
        partition = label = region_id = start = end = ""
    else:
        kind = "instance"
        parts = PurePosixPath(name).parts
        partition, label = parts[0], parts[1]
        match = WINDOW_RE.search(parts[-1])
        if match is None:
            raise ValueError(f"Cannot parse instance filename: {name}")
        region_id = match["region"]
        start, end = match["start"], match["end"]
    return dict(
        archive=archive.name, member=name, path=relative, kind=kind,
        partition=partition, label=label, region_id=region_id,
        start=start, end=end, bytes=member.size,
    )


def main() -> None:
    missing = [str(p) for p in ARCHIVES if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing archives: " + ", ".join(missing))
    DATA.mkdir(exist_ok=True)
    manifest = DATA / "manifest.csv"
    status = DATA / "extraction_status.json"
    counts: dict[str, int] = {}
    seen: set[str] = set()
    with manifest.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for archive in ARCHIVES:
            count = 0
            written = 0
            with tarfile.open(archive, "r|gz") as source:
                for member in source:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError(f"Unsupported archive entry: {member.name}")
                    target, relative = destination(archive, member)
                    if relative in seen:
                        raise ValueError(f"Path collision: {relative}")
                    seen.add(relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or target.stat().st_size != member.size:
                        if count % 1000 == 0 and shutil.disk_usage(DATA).free < MIN_FREE_BYTES:
                            raise OSError("Extraction stopped: less than 750 MB free on D:")
                        stream = source.extractfile(member)
                        if stream is None:
                            raise OSError(f"Cannot read {member.name}")
                        with stream, target.open("wb") as dest:
                            shutil.copyfileobj(stream, dest, 1024 * 1024)
                        written += 1
                    writer.writerow(manifest_row(archive, member, relative))
                    count += 1
            counts[archive.name] = count
            status.write_text(json.dumps({"completed_archives": counts}, indent=2), encoding="utf-8")
            out.flush()
            print(f"{archive.name}: {count:,} files ({written:,} written)", flush=True)
    status.write_text(json.dumps({"completed_archives": counts, "complete": True}, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
