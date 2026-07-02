from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator


ORATS_FILE_RE = re.compile(r"^ORATS_SMV_Strikes_(\d{8})\.csv$")


@dataclass(frozen=True)
class OratsFile:
    path: Path
    file_date: date
    year: int
    size_bytes: int
    header: tuple[str, ...]
    schema_hash: str
    row_count: int | None = None

    @property
    def schema_version(self) -> str:
        return self.schema_hash[:12]


def parse_file_date(path: Path) -> date | None:
    match = ORATS_FILE_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def schema_hash(header: Iterable[str]) -> str:
    normalized = ",".join(col.strip() for col in header)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_header(path: Path) -> tuple[str, ...]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        return tuple(next(reader))


def count_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


def iter_orats_csv_files(source_root: Path) -> Iterator[Path]:
    for path in sorted(source_root.glob("*/*.csv")):
        if path.name.startswith("._"):
            continue
        if parse_file_date(path) is None:
            continue
        yield path


def discover_files(
    source_root: Path,
    start: date | None = None,
    end: date | None = None,
    include_row_counts: bool = False,
) -> list[OratsFile]:
    files: list[OratsFile] = []
    for path in iter_orats_csv_files(source_root):
        file_date = parse_file_date(path)
        if file_date is None:
            continue
        if start is not None and file_date < start:
            continue
        if end is not None and file_date > end:
            continue
        header = read_header(path)
        row_count = count_data_rows(path) if include_row_counts else None
        files.append(
            OratsFile(
                path=path,
                file_date=file_date,
                year=file_date.year,
                size_bytes=path.stat().st_size,
                header=header,
                schema_hash=schema_hash(header),
                row_count=row_count,
            )
        )
    return files


def write_inventory_csv(files: list[OratsFile], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "file_date",
        "year",
        "size_bytes",
        "schema_hash",
        "schema_version",
        "row_count",
        "columns",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in files:
            writer.writerow(
                {
                    "path": str(item.path),
                    "file_date": item.file_date.isoformat(),
                    "year": item.year,
                    "size_bytes": item.size_bytes,
                    "schema_hash": item.schema_hash,
                    "schema_version": item.schema_version,
                    "row_count": "" if item.row_count is None else item.row_count,
                    "columns": "|".join(item.header),
                }
            )
