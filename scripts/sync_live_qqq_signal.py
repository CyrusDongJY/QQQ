from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "src/qqq_options_system/live/qqq_strategy_email.py"
SCANNER_SOURCE = PROJECT_ROOT / "src/qqq_options_system/live/qqq_contract_scanner.py"
CLOUD_ROOT = (
    PROJECT_ROOT / "cloud_integration"
    if (PROJECT_ROOT / "cloud_integration").exists()
    else PROJECT_ROOT.parent / "cloud_integration"
)
DEFAULT_DESTINATION = CLOUD_ROOT / "qqq_strategy_email.py"
DEFAULT_SCANNER_DESTINATION = CLOUD_ROOT / "qqq_contract_scanner.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync(source: Path, destination: Path, check_only: bool = False) -> bool:
    destination = destination.expanduser().resolve()
    matches = destination.exists() and sha256(destination) == sha256(source)
    if check_only:
        state = "OK" if matches else "OUT_OF_SYNC"
        print(f"{state} {destination}")
        return matches
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_hash = sha256(destination)
    if copied_hash != sha256(source):
        raise RuntimeError(f"Hash mismatch after copying to {destination}")
    print(f"SYNCED {destination} sha256={copied_hash}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync the authoritative QQQ email-signal module to vendored copies."
    )
    parser.add_argument(
        "destinations",
        nargs="*",
        type=Path,
        default=[DEFAULT_DESTINATION],
    )
    parser.add_argument(
        "--scanner-destination",
        type=Path,
        default=DEFAULT_SCANNER_DESTINATION,
        help="Destination for qqq_contract_scanner.py.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    results = [
        sync(SOURCE, path, check_only=args.check) for path in args.destinations
    ]
    results.append(
        sync(
            SCANNER_SOURCE,
            args.scanner_destination,
            check_only=args.check,
        )
    )
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
