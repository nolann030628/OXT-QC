from __future__ import annotations

from pathlib import Path


def check_dataset(input_dir: str) -> dict:
    base = Path(input_dir)
    exists = base.exists() and base.is_dir()
    file_count = len([p for p in base.rglob("*") if p.is_file()]) if exists else 0
    return {
        "input_dir": str(base),
        "exists": exists,
        "file_count": file_count,
        "status": "ok" if exists else "missing_input_dir",
    }
