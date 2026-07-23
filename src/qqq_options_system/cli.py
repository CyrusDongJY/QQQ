from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from qqq_options_system.data.audit import audit_qqq_file, write_daily_audit_csv
from qqq_options_system.data.normalize import normalize_qqq_wide_to_long, write_partitioned_parquet
from qqq_options_system.data.orats_files import discover_files, write_inventory_csv
from qqq_options_system.research.experiments import load_experiment_config, run_experiment
from qqq_options_system.research.regime_experiment import load_regime_config, run_regime_study
from qqq_options_system.research.walk_forward import run_walk_forward_experiment


DEFAULT_CONFIG = Path("configs/qqq_pilot.json")


def parse_date(text: str):
    return datetime.strptime(text, "%Y-%m-%d").date()


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--underlying", default=None)


def resolve_settings(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    return {
        "source_root": Path(args.source_root or config.get("source_root", ".")),
        "output_root": Path(args.output_root or config.get("output_root", "outputs")),
        "underlying": (args.underlying or config.get("underlying", "QQQ")).upper(),
        "chunk_size": int(getattr(args, "chunk_size", None) or config.get("chunk_size", 200_000)),
        "pilot_start": config.get("pilot_start", "2026-06-01"),
        "pilot_end": config.get("pilot_end", "2026-06-30"),
    }


def cmd_inventory(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    start = parse_date(args.start) if args.start else None
    end = parse_date(args.end) if args.end else None
    output_path = args.output or (
        settings["output_root"] / "data" / "raw_manifest" / "orats_inventory.csv"
    )
    files = discover_files(
        settings["source_root"],
        start=start,
        end=end,
        include_row_counts=args.row_count,
    )
    write_inventory_csv(files, output_path)
    print(f"Wrote {len(files)} inventory rows to {output_path}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    start = parse_date(args.start or settings["pilot_start"])
    end = parse_date(args.end or settings["pilot_end"])
    output_path = args.output or (
        settings["output_root"]
        / "reports"
        / f"qqq_daily_audit_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    )
    files = discover_files(settings["source_root"], start=start, end=end)
    rows = [
        audit_qqq_file(item, underlying=settings["underlying"], chunk_size=settings["chunk_size"])
        for item in files
    ]
    write_daily_audit_csv(rows, output_path)
    print(f"Wrote {len(rows)} audit rows to {output_path}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    if args.year is not None:
        start = parse_date(f"{args.year}-01-01")
        end = parse_date(f"{args.year}-12-31")
    else:
        start = parse_date(args.start or settings["pilot_start"])
        end = parse_date(args.end or settings["pilot_end"])
    files = discover_files(settings["source_root"], start=start, end=end)
    written: list[Path] = []
    skipped = 0
    for item in files:
        output_path = (
            settings["output_root"]
            / "data"
            / "silver_options"
            / f"symbol={settings['underlying']}"
            / f"year={item.file_date.year}"
            / f"trade_date={item.file_date.isoformat()}"
            / "option_chain_eod.parquet"
        )
        if output_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
            print(f"Skipping existing partition {output_path}", flush=True)
            continue
        normalized_chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(item.path, chunksize=settings["chunk_size"], low_memory=False):
            normalized = normalize_qqq_wide_to_long(
                chunk,
                source_file=str(item.path),
                schema_version=item.schema_version,
                underlying=settings["underlying"],
            )
            if not normalized.empty:
                normalized_chunks.append(normalized)
        if not normalized_chunks:
            print(f"No {settings['underlying']} rows in {item.path}")
            continue
        day_frame = pd.concat(normalized_chunks, ignore_index=True)
        output_path = write_partitioned_parquet(
            day_frame,
            settings["output_root"],
            trade_date=item.file_date,
            underlying=settings["underlying"],
        )
        written.append(output_path)
        print(f"Wrote {len(day_frame)} normalized rows to {output_path}", flush=True)
    print(f"Wrote {len(written)} parquet partitions; skipped {skipped}")
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    config = load_experiment_config(args.experiment_config)
    if args.data_start:
        config["data_start"] = args.data_start
    if args.data_end:
        config["data_end"] = args.data_end
    if args.research_start:
        config["research_start"] = args.research_start
    if args.research_end:
        config["research_end"] = args.research_end
    if args.entry_lag_days is not None:
        config["entry_lag_days"] = args.entry_lag_days
    config["underlying"] = settings["underlying"]
    paths = run_experiment(config, settings["output_root"])
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_run_walk_forward(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    config = load_experiment_config(args.experiment_config)
    if args.data_start:
        config["data_start"] = args.data_start
    if args.data_end:
        config["data_end"] = args.data_end
    if args.entry_lag_days is not None:
        config["entry_lag_days"] = args.entry_lag_days
    config["underlying"] = settings["underlying"]
    paths = run_walk_forward_experiment(config, settings["output_root"])
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_run_regime_study(args: argparse.Namespace) -> int:
    settings = resolve_settings(args)
    config = load_regime_config(args.regime_config)
    if args.data_start:
        config["data_start"] = args.data_start
    if args.data_end:
        config["data_end"] = args.data_end
    config["underlying"] = settings["underlying"]
    paths = run_regime_study(config, settings["output_root"])
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qqq-options")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inventory ORATS CSV files")
    add_common_args(inventory)
    inventory.add_argument("--start")
    inventory.add_argument("--end")
    inventory.add_argument("--row-count", action="store_true")
    inventory.add_argument("--output", type=Path)
    inventory.set_defaults(func=cmd_inventory)

    audit = subparsers.add_parser("audit", help="Audit QQQ rows by day")
    add_common_args(audit)
    audit.add_argument("--start")
    audit.add_argument("--end")
    audit.add_argument("--chunk-size", type=int)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(func=cmd_audit)

    convert = subparsers.add_parser("convert", help="Convert QQQ rows to normalized parquet")
    add_common_args(convert)
    convert.add_argument("--start")
    convert.add_argument("--end")
    convert.add_argument("--year", type=int, help="Convert a single calendar year")
    convert.add_argument("--chunk-size", type=int)
    convert.add_argument("--skip-existing", action="store_true")
    convert.add_argument("--overwrite", action="store_true")
    convert.set_defaults(func=cmd_convert)

    experiment = subparsers.add_parser("run-experiment", help="Run trigger/option parameter research")
    add_common_args(experiment)
    experiment.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/research_sample.json"),
    )
    experiment.add_argument("--data-start")
    experiment.add_argument("--data-end")
    experiment.add_argument("--research-start")
    experiment.add_argument("--research-end")
    experiment.add_argument("--entry-lag-days", type=int)
    experiment.set_defaults(func=cmd_run_experiment)

    walk_forward = subparsers.add_parser(
        "run-walk-forward",
        help="Run training-only ranked walk-forward research",
    )
    add_common_args(walk_forward)
    walk_forward.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/research_walk_forward.json"),
    )
    walk_forward.add_argument("--data-start")
    walk_forward.add_argument("--data-end")
    walk_forward.add_argument("--entry-lag-days", type=int)
    walk_forward.set_defaults(func=cmd_run_walk_forward)

    regime = subparsers.add_parser(
        "run-regime-study",
        help="Run five-factor VIX/VXN regime cohort study",
    )
    add_common_args(regime)
    regime.add_argument(
        "--regime-config",
        type=Path,
        default=Path("configs/regime_vix_pilot.json"),
    )
    regime.add_argument("--data-start")
    regime.add_argument("--data-end")
    regime.set_defaults(func=cmd_run_regime_study)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
