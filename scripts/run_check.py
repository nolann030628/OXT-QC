#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.report import write_report
from src.vlm_checker import check_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tactile data completeness checks.")
    parser.add_argument("--input", required=True, help="Input tactile dataset directory.")
    parser.add_argument("--config", required=False, help="Path to config file.")
    parser.add_argument("--output", required=True, help="Output report path.")
    args = parser.parse_args()

    report = check_dataset(args.input)
    if args.config:
        report["config"] = args.config

    write_report(report, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
