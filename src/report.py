from __future__ import annotations

import json
from pathlib import Path


def write_report(report: dict, output_file: str) -> None:
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
