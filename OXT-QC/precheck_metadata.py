#!/usr/bin/env python3
"""openxtactile_check_zarr.py

Open-X-Tactile (OXT) zarr dataset **precheck & metadata export** tool.

Given a zarr dataset (or a directory that contains several ``*.zarr`` datasets)
stored in the OXT replay-buffer layout::

    <dataset>.zarr/
    ├── data/
    │   ├── timestamps                  (T,)
    │   ├── <side>_wrist_camera_rgb     (T, H, W, 3)   # or camera_ego_rgb ...
    │   ├── <side>_hand_joints          (T, N)
    │   ├── <side>_hand_joints_idx      (T, N)
    │   ├── <side>_tactile_data_<key>   (T, C, ...)    # four-piece set
    │   ├── <side>_tactile_area_<key>   (T, C)
    │   ├── <side>_tactile_sensor_<key> (T,)
    │   ├── <side>_tactile_type_<key>   (T,)
    │   └── ...
    └── meta/
        └── episode_ends                (num_episodes,)

this tool performs the following checks (see ``--help``):

  1. the zarr can be opened and exposes ``data`` / ``meta``;
  2. records ``key / shape / dtype`` for every array and verifies the frame
     dimension ``T`` is consistent across all data arrays;
  3. ``timestamps`` exist, are monotonically increasing **per episode**, and
     have no unexpected gaps (断层);
  4. at least one RGB stream is present;
  5. every tactile "four-piece set" (``data`` / ``area`` / ``sensor`` /
     ``type``) is complete;
  6. ``hand_joints`` and ``hand_joints_idx`` are paired and shape-consistent;
  7. optional ``supplementary_joints`` / ``supplementary_tactile`` presence;

and finally writes to the output directory:

  * ``precheck_report.json`` -- the *summary* file: pass / warn / fail details
    for every check **plus** each dataset's technical metadata (keys, shapes,
    dtypes, timestamps, tactile sets, hand joints, ...);
  * ``metadata.json`` -- a **website-renderable** descriptor using the exact
    schema of ``sources/<name>/metadata.json``. All ``*.zarr`` under the input
    path are treated as one data source and aggregated; every auto-derivable
    field (``task`` / ``trajectory`` / ``frame`` / ``sensor_*`` / ``type``) is
    filled in. Human-only fields (``url`` / ``description`` / ``contact`` /
    ``reference`` / ...) are left as placeholders, or copied from an existing
    file via ``--merge-metadata``. Fields that were guessed are listed under
    ``metadata_export.needs_review`` in ``precheck_report.json``.

The process exit code is non-zero when any dataset fails a hard check, so the
tool can be dropped into CI / contributor pipelines.

Example
-------
    # validate + export a renderable metadata.json for the whole source
    python openxtactile_check_zarr.py /path/to/ViTaMIn \
        --output-dir ./oxt_check_output --source-name ViTaMIn

    # refresh counts while preserving the human-authored fields
    python openxtactile_check_zarr.py /path/to/ViTaMIn \
        --merge-metadata sources/ViTaMIn/metadata.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zarr
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "This tool requires the 'zarr' package. Install it with:\n"
        "    pip install zarr numpy\n"
        f"(import error: {exc})"
    )

from imagecodecs.numcodecs import register_codecs

register_codecs(force=False, verbose=False)

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# Constants describing the OXT standard layout.
# --------------------------------------------------------------------------- #

# Keys whose presence marks an RGB image stream.
RGB_KEY_SUFFIX = "_rgb"

# The four members that together make up a tactile "four-piece set" (四件套).
TACTILE_MEMBERS = ("data", "area", "sensor", "type")

# Regex splitting a tactile key into (prefix, member, detail).
#   e.g. "right_tactile_data_gripper" -> ("right", "data", "gripper")
_TACTILE_RE = re.compile(
    r"^(?P<prefix>.+)_tactile_(?P<member>data|area|sensor|type)_(?P<detail>.+)$"
)

# Allowed values for the tactile ``type`` stream.
VALID_TACTILE_TYPES = {"state", "binary", "image", "matrix"}

# Optional / supplementary keys we probe for.
SUPPLEMENTARY_KEYS = ("supplementary_joints", "supplementary_tactile")

# Factor above the median per-step delta that counts as a timestamp gap (断层).
DEFAULT_GAP_FACTOR = 3.0

# zarr tactile ``type`` -> website sensor-kind bucket (image / array / state).
# 'image' is unambiguous; the state/binary -> state/array split is a best-effort
# heuristic (e.g. taxel arrays vs force/torque state) that a human should verify.
TACTILE_TYPE_TO_SENSOR_KIND = {
    "image": "image",
    "state": "state",
    "binary": "array",
    "matrix": "array",
}

# Website metadata fields that cannot be derived from the zarr (human-authored).
HUMAN_METADATA_FIELDS = (
    "name", "url", "version", "description", "contact", "reference", "code_trigger", "label",
)

# Status levels for individual checks.
PASS = "pass"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


# Keys matching these patterns are expected to stay constant over time.
CONSTANT_KEY_SUBSTRINGS = (
    "sensor_type",
    "sensor_name",
    "joint_idx",
    "func_area_idx",
    "_tactile_area_",
    "_tactile_sensor_",
    "_tactile_type_",
)


# --------------------------------------------------------------------------- #
# Small result helpers.
# --------------------------------------------------------------------------- #

@dataclass
class Check:
    """A single named check result."""

    name: str
    status: str
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "status": self.status}
        if self.message:
            out["message"] = self.message
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class DatasetReport:
    """Accumulates checks for one zarr dataset."""

    name: str
    path: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, message: str = "", **detail: Any) -> None:
        self.checks.append(Check(name, status, message, detail))

    @property
    def num_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == FAIL)

    @property
    def num_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == WARN)

    @property
    def passed(self) -> bool:
        return self.num_failed == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "passed": self.passed,
            "num_errors": self.num_failed,
            "num_warnings": self.num_warn,
            "checks": [c.to_dict() for c in self.checks],
        }


# --------------------------------------------------------------------------- #
# JSON helpers (numpy types are not natively serialisable).
# --------------------------------------------------------------------------- #

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val):
            return None
        return val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def _shape(arr: Any) -> list[int]:
    return [int(x) for x in arr.shape]


def _format_json_compact_lists(obj: Any, indent: int = 2, level: int = 0) -> str:
    pad = " " * (indent * level)
    next_pad = " " * (indent * (level + 1))

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for key, value in obj.items():
            rendered = _format_json_compact_lists(value, indent=indent, level=level + 1)
            items.append(f'{next_pad}{json.dumps(str(key), ensure_ascii=False)}: {rendered}')
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"

    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in obj):
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
        items = [_format_json_compact_lists(x, indent=indent, level=level + 1) for x in obj]
        return "[\n" + ",\n".join(f"{next_pad}{item}" for item in items) + "\n" + pad + "]"

    return json.dumps(obj, ensure_ascii=False)


def _is_constant_candidate_key(key: str) -> bool:
    lower = key.lower()
    if any(token in lower for token in CONSTANT_KEY_SUBSTRINGS):
        return True
    if lower.endswith("_idx") or "_idx_" in lower:
        return True
    return False


def _values_equal_with_nan(arr: np.ndarray, first: np.ndarray) -> np.ndarray:
    eq = (arr == first)
    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(first.dtype, np.floating):
        eq = eq | (np.isnan(arr) & np.isnan(first))
    return eq


def _check_constant_like_keys(
    data: Any,
    data_keys: list[str],
    report: DatasetReport,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    changed_keys: list[str] = []

    for key in data_keys:
        if not _is_constant_candidate_key(key):
            continue

        arr = np.asarray(data[key][:])
        entry: dict[str, Any] = {
            "expected_constant": True,
            "shape": _shape(data[key]),
            "dtype": str(data[key].dtype),
        }

        if arr.ndim == 0:
            entry["is_constant_over_time"] = True
            entry["constant_value"] = _to_jsonable(arr)
            summary[key] = entry
            continue

        if arr.shape[0] == 0:
            entry["is_constant_over_time"] = True
            entry["constant_value"] = None
            summary[key] = entry
            continue

        first = arr[0]
        if arr.ndim == 1:
            row_equal = _values_equal_with_nan(arr, first)
        else:
            axes = tuple(range(1, arr.ndim))
            row_equal = np.all(_values_equal_with_nan(arr, first), axis=axes)

        is_constant = bool(np.all(row_equal))
        entry["is_constant_over_time"] = is_constant
        if is_constant:
            entry["constant_value"] = _to_jsonable(first)
        else:
            changed_keys.append(key)
            first_change_index = int(np.flatnonzero(~row_equal)[0])
            entry["first_value"] = _to_jsonable(first)
            entry["first_change_index"] = first_change_index
            entry["num_changed_frames"] = int(np.sum(~row_equal))
            report.add(
                f"constant_like::{key}",
                WARN,
                f"'{key}' looks like a constant/index key but changes over time",
                first_change_index=first_change_index,
                num_changed_frames=int(np.sum(~row_equal)),
            )
        summary[key] = entry

    if not summary:
        report.add("constant_like_keys", SKIP, "no constant-like keys found")
    elif changed_keys:
        report.add(
            "constant_like_keys",
            WARN,
            f"{len(changed_keys)} constant-like key(s) change over time",
            changed_keys=changed_keys,
            num_checked=len(summary),
        )
    else:
        report.add(
            "constant_like_keys",
            PASS,
            f"all {len(summary)} constant-like key(s) are stable over time",
            checked_keys=sorted(summary.keys()),
        )

    return summary


def _build_dict_export(
    key_info: dict[str, dict[str, Any]],
    meta_key_info: dict[str, dict[str, Any]],
    constant_like: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, info in key_info.items():
        entry: dict[str, Any] = {
            "shape": info["shape"],
            "dtype": info["dtype"],
        }
        if "compression" in info:
            entry["compression"] = info["compression"]
        if key in constant_like:
            extra = {k: v for k, v in constant_like[key].items() if k not in ("shape", "dtype")}
            entry.update(extra)
        out[key] = entry

    for key, info in meta_key_info.items():
        out[f"meta/{key}"] = {
            "shape": info["shape"],
            "dtype": info["dtype"],
        }
    return out


# --------------------------------------------------------------------------- #
# Dataset discovery.
# --------------------------------------------------------------------------- #

def _is_zarr_dir(path: Path) -> bool:
    """A directory is a zarr root if it holds a v2 ``.zgroup``/``.zarray`` or a
    v3 ``zarr.json`` marker."""
    if not path.is_dir():
        return False
    for marker in (".zgroup", ".zarray", "zarr.json"):
        if (path / marker).exists():
            return True
    return False


def discover_datasets(path: Path, pattern: str = "*.zarr") -> list[Path]:
    """Return the list of zarr datasets to check.

    ``path`` may be a single zarr dataset or a directory that contains several
    of them (matched by ``pattern``, defaulting to ``*.zarr``).
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")

    if _is_zarr_dir(path):
        return [path]

    if path.is_dir():
        found = sorted(p for p in path.glob(pattern) if _is_zarr_dir(p))
        if found:
            return found
        # Fall back: maybe the directory itself is meant to be a zarr but is
        # missing markers; let the open step report the real error.
        return [path]

    raise ValueError(f"unsupported path (not a dir): {path}")


# --------------------------------------------------------------------------- #
# Tactile-key parsing.
# --------------------------------------------------------------------------- #

def _group_tactile_keys(keys: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Group tactile keys into four-piece sets.

    Returns ``{(prefix, detail): {member: key}}``.
    """
    sets: dict[tuple[str, str], dict[str, str]] = {}
    for key in keys:
        m = _TACTILE_RE.match(key)
        if not m:
            continue
        set_id = (m.group("prefix"), m.group("detail"))
        sets.setdefault(set_id, {})[m.group("member")] = key
    return sets


# --------------------------------------------------------------------------- #
# Core per-dataset check.
# --------------------------------------------------------------------------- #

def check_dataset(zarr_path: Path, gap_factor: float = DEFAULT_GAP_FACTOR) -> tuple[DatasetReport, dict[str, Any]]:
    """Run all checks on a single zarr dataset.

    Returns ``(report, metadata_summary)``.
    """
    name = zarr_path.name
    report = DatasetReport(name=name, path=str(zarr_path))
    meta_summary: dict[str, Any] = {"name": name, "path": str(zarr_path)}

    # --- 1. open ---------------------------------------------------------- #
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
    except Exception as exc:  # noqa: BLE001 - report any open failure
        report.add(
            "open_zarr", FAIL,
            f"failed to open zarr: {type(exc).__name__}: {exc}",
        )
        meta_summary["error"] = f"{type(exc).__name__}: {exc}"
        return report, meta_summary
    report.add("open_zarr", PASS, "zarr group opened successfully")

    # Locate the data group (standard: root['data']; fallback: arrays at root).
    if "data" in root:
        data = root["data"]
        report.add("has_data_group", PASS, "'data' group present")
    else:
        data = root
        report.add(
            "has_data_group", WARN,
            "no 'data' subgroup; treating root arrays as data",
        )

    data_keys = sorted(data.array_keys())
    if not data_keys:
        report.add("data_arrays", FAIL, "no arrays found under data group")
        meta_summary["error"] = "no data arrays"
        return report, meta_summary

    # --- meta / episode_ends --------------------------------------------- #
    episode_ends: np.ndarray | None = None
    meta_grp = root["meta"] if "meta" in root else None
    if meta_grp is not None and "episode_ends" in meta_grp.array_keys():
        episode_ends = np.asarray(meta_grp["episode_ends"][:]).astype(np.int64)
        n_ep = int(episode_ends.shape[0])
        ok_mono = bool(np.all(np.diff(episode_ends) > 0)) if n_ep > 1 else True
        if n_ep == 0:
            report.add("episode_ends", FAIL, "episode_ends is empty")
        elif not ok_mono:
            report.add(
                "episode_ends", FAIL,
                "episode_ends is not strictly increasing",
                num_episodes=n_ep,
            )
        else:
            report.add(
                "episode_ends", PASS,
                f"{n_ep} episode(s)", num_episodes=n_ep,
                last=int(episode_ends[-1]),
            )
    else:
        report.add(
            "episode_ends", WARN,
            "meta/episode_ends missing; episode-aware checks are limited",
        )

    # --- 2. keys / shape / dtype + frame consistency --------------------- #
    key_info: dict[str, dict[str, Any]] = {}
    frame_counts: dict[str, int] = {}
    for k in data_keys:
        arr = data[k]
        shp = _shape(arr)
        key_info[k] = {"shape": shp, "dtype": str(arr.dtype)}
        if k.endswith((RGB_KEY_SUFFIX, "_depth", "_image")) or len(shp) >= 4:
            if arr.metadata.zarr_format == 2:
                compressor = arr.metadata.compressor
                key_info[k]["compression"] = (
                    compressor.get_config() if compressor is not None else None
                )
            else:
                key_info[k]["compression"] = [
                    codec.to_dict() for codec in arr.compressors
                ]
        if shp:
            frame_counts[k] = shp[0]
    report.add(
        "key_shape_dtype", PASS,
        f"catalogued {len(key_info)} data array(s)",
        num_keys=len(key_info),
    )

    # Frame-count consistency: every data array must share the same T (axis 0).
    unique_T = sorted(set(frame_counts.values()))
    T: int | None = None
    if len(unique_T) == 1:
        T = unique_T[0]
        report.add("frame_consistency", PASS, f"all arrays share T={T}", num_frames=T)
    else:
        # Pick the majority T for downstream reporting.
        counts: dict[int, int] = {}
        for v in frame_counts.values():
            counts[v] = counts.get(v, 0) + 1
        T = max(counts, key=counts.get)
        mismatched = {k: frame_counts[k] for k in frame_counts if frame_counts[k] != T}
        report.add(
            "frame_consistency", FAIL,
            f"inconsistent frame dim across arrays; majority T={T}",
            majority_T=T, mismatched=mismatched,
        )

    # episode_ends[-1] should equal T.
    if episode_ends is not None and episode_ends.size and T is not None:
        if int(episode_ends[-1]) != T:
            report.add(
                "episode_ends_vs_frames", FAIL,
                f"episode_ends[-1]={int(episode_ends[-1])} != T={T}",
            )
        else:
            report.add("episode_ends_vs_frames", PASS, f"episode_ends[-1] == T == {T}")

    # --- 3. timestamps ---------------------------------------------------- #
    ts_summary = _check_timestamps(data, data_keys, episode_ends, report, gap_factor)

    # --- 4. RGB ----------------------------------------------------------- #
    rgb_keys = [k for k in data_keys if k.endswith(RGB_KEY_SUFFIX)]
    if rgb_keys:
        bad_rgb = []
        for k in rgb_keys:
            shp = key_info[k]["shape"]
            dt = key_info[k]["dtype"]
            if len(shp) != 4 or shp[-1] not in (3, 4) or dt != "uint8":
                bad_rgb.append({"key": k, "shape": shp, "dtype": dt})
        if bad_rgb:
            report.add(
                "rgb_present", WARN,
                f"found {len(rgb_keys)} RGB key(s) but some look non-standard "
                "(expected [T,H,W,3] uint8)",
                rgb_keys=rgb_keys, suspicious=bad_rgb,
            )
        else:
            report.add(
                "rgb_present", PASS,
                f"found {len(rgb_keys)} RGB stream(s)", rgb_keys=rgb_keys,
            )
    else:
        report.add("rgb_present", FAIL, "no '*_rgb' image stream found")

    # --- 5. tactile four-piece set --------------------------------------- #
    tactile_summary = _check_tactile(data, data_keys, key_info, report)

    # --- 6. hand_joints / hand_joints_idx -------------------------------- #
    hand_summary = _check_hand_joints(data_keys, key_info, report)

    # --- 7. constant-like keys ------------------------------------------- #
    constant_like_summary = _check_constant_like_keys(data, data_keys, report)

    # --- 8. supplementary ------------------------------------------------- #
    supp_summary: dict[str, Any] = {}
    for sk in SUPPLEMENTARY_KEYS:
        if sk in key_info:
            supp_summary[sk] = key_info[sk]
            report.add(
                f"supplementary::{sk}", PASS,
                f"'{sk}' present", shape=key_info[sk]["shape"],
                dtype=key_info[sk]["dtype"],
            )
        else:
            supp_summary[sk] = None
            report.add(f"supplementary::{sk}", SKIP, f"'{sk}' not present (optional)")

    # --- assemble metadata summary --------------------------------------- #
    meta_key_info = (
        {k: {"shape": _shape(meta_grp[k]), "dtype": str(meta_grp[k].dtype)}
         for k in sorted(meta_grp.array_keys())}
        if meta_grp is not None else {}
    )

    meta_summary.update({
        "num_episodes": int(episode_ends.shape[0]) if episode_ends is not None else None,
        "num_frames": T,
        "num_data_keys": len(key_info),
        "keys": key_info,
        "meta_keys": meta_key_info,
        "timestamps": ts_summary,
        "rgb_keys": rgb_keys,
        "tactile_sets": tactile_summary,
        "hand_joints": hand_summary,
        "constant_like_keys": constant_like_summary,
        "supplementary": supp_summary,
        "instructions": _instruction_samples(data, data_keys),
        "dict_export": _build_dict_export(key_info, meta_key_info, constant_like_summary),
        "precheck_passed": report.passed,
    })

    return report, meta_summary


# --------------------------------------------------------------------------- #
# Check sub-routines.
# --------------------------------------------------------------------------- #

def _check_timestamps(
    data: Any,
    data_keys: list[str],
    episode_ends: np.ndarray | None,
    report: DatasetReport,
    gap_factor: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"present": False}
    if "timestamps" not in data_keys:
        report.add("timestamps_present", FAIL, "'timestamps' key is missing")
        return summary

    ts = np.asarray(data["timestamps"][:]).reshape(-1)
    summary.update({
        "present": True,
        "dtype": str(data["timestamps"].dtype),
        "count": int(ts.shape[0]),
        "min": _to_jsonable(ts.min()) if ts.size else None,
        "max": _to_jsonable(ts.max()) if ts.size else None,
    })
    report.add("timestamps_present", PASS, f"timestamps present ({ts.shape[0]} values)")

    if ts.size == 0:
        report.add("timestamps_monotonic", FAIL, "timestamps array is empty")
        return summary

    # Build per-episode slices; fall back to the whole array as one episode.
    if episode_ends is not None and episode_ends.size:
        bounds = np.concatenate([[0], episode_ends.astype(np.int64)])
        slices = [slice(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]
    else:
        slices = [slice(0, ts.size)]

    non_increasing = 0
    all_diffs: list[np.ndarray] = []
    for sl in slices:
        seg = ts[sl]
        if seg.size < 2:
            continue
        d = np.diff(seg.astype(np.float64))
        all_diffs.append(d)
        non_increasing += int(np.sum(d <= 0))

    global_mono = bool(np.all(np.diff(ts.astype(np.float64)) > 0)) if ts.size > 1 else True
    summary["global_monotonic"] = global_mono

    if non_increasing == 0:
        summary["per_episode_monotonic"] = True
        report.add(
            "timestamps_monotonic", PASS,
            "timestamps strictly increasing within every episode"
            + ("" if global_mono else " (reset at episode boundaries, as expected)"),
        )
    else:
        summary["per_episode_monotonic"] = False
        report.add(
            "timestamps_monotonic", FAIL,
            f"{non_increasing} non-increasing step(s) found within episodes",
            num_non_increasing=non_increasing,
        )

    # Gap (断层) detection based on per-episode positive deltas.
    if all_diffs:
        diffs = np.concatenate(all_diffs)
        pos = diffs[diffs > 0]
        if pos.size:
            median_step = float(np.median(pos))
            threshold = median_step * gap_factor
            gaps = int(np.sum(diffs > threshold)) if threshold > 0 else 0
            summary.update({
                "median_step": median_step,
                "max_step": _to_jsonable(diffs.max()),
                "gap_threshold": threshold,
                "num_gaps": gaps,
            })
            if gaps == 0:
                report.add(
                    "timestamps_gaps", PASS,
                    f"no gaps (median step={median_step:g}, max step={diffs.max():g})",
                )
            else:
                report.add(
                    "timestamps_gaps", WARN,
                    f"{gaps} gap(s) exceed {gap_factor}x median step "
                    f"({median_step:g}); max step={diffs.max():g}",
                    num_gaps=gaps, median_step=median_step,
                    max_step=_to_jsonable(diffs.max()),
                )
    return summary


def _check_tactile(
    data: Any,
    data_keys: list[str],
    key_info: dict[str, dict[str, Any]],
    report: DatasetReport,
) -> list[dict[str, Any]]:
    tactile_sets = _group_tactile_keys(data_keys)
    if not tactile_sets:
        report.add("tactile_present", WARN, "no tactile keys found in dataset")
        return []

    summaries: list[dict[str, Any]] = []
    for (prefix, detail), members in sorted(tactile_sets.items()):
        set_label = f"{prefix}/{detail}"
        missing = [m for m in TACTILE_MEMBERS if m not in members]
        set_summary: dict[str, Any] = {
            "set": set_label,
            "prefix": prefix,
            "detail": detail,
            "members_present": sorted(members.keys()),
            "members_missing": missing,
            "complete": not missing,
        }

        if missing:
            report.add(
                f"tactile_set::{set_label}", FAIL,
                f"incomplete four-piece set, missing: {', '.join(missing)}",
                present=sorted(members.keys()), missing=missing,
            )
        else:
            # Record shapes and sanity-check channel alignment.
            data_shape = key_info[members["data"]]["shape"]
            area_shape = key_info[members["area"]]["shape"]
            set_summary["data_shape"] = data_shape
            set_summary["data_dtype"] = key_info[members["data"]]["dtype"]
            set_summary["area_shape"] = area_shape

            issues = []
            # data: [T, C, ...]; area: [T, C]; sensor/type: [T]
            n_channels = data_shape[1] if len(data_shape) >= 2 else None
            set_summary["num_channels"] = n_channels
            if len(area_shape) >= 2 and n_channels is not None and area_shape[1] != n_channels:
                issues.append(
                    f"channel mismatch: data C={n_channels} vs area C={area_shape[1]}"
                )

            # Unique sensor names + tactile types (small string streams).
            try:
                sensor_vals = np.asarray(data[members["sensor"]][:]).astype(str)
                set_summary["sensor"] = sorted(set(sensor_vals.tolist()))
            except Exception:  # noqa: BLE001
                set_summary["sensor"] = None
            try:
                type_vals = np.asarray(data[members["type"]][:]).astype(str)
                uniq_types = sorted(set(type_vals.tolist()))
                set_summary["type"] = uniq_types
                bad_types = [t for t in uniq_types if t not in VALID_TACTILE_TYPES]
                if bad_types:
                    issues.append(f"unknown tactile type(s): {bad_types}")
            except Exception:  # noqa: BLE001
                set_summary["type"] = None

            if issues:
                report.add(
                    f"tactile_set::{set_label}", WARN,
                    "complete but with issues: " + "; ".join(issues),
                    **set_summary,
                )
            else:
                report.add(
                    f"tactile_set::{set_label}", PASS,
                    f"complete four-piece set (C={n_channels}, "
                    f"type={set_summary.get('type')})",
                )
        summaries.append(set_summary)
    return summaries


def _check_hand_joints(
    data_keys: list[str],
    key_info: dict[str, dict[str, Any]],
    report: DatasetReport,
) -> dict[str, Any]:
    # A "hand_joints" key looks like "<side>_hand_joints" (exclude *_idx / *_action).
    joint_keys = [
        k for k in data_keys
        if k.endswith("_hand_joints") or k == "hand_joints"
    ]
    summary: dict[str, Any] = {}
    if not joint_keys:
        report.add("hand_joints", WARN, "no '*_hand_joints' key found")
        return summary

    for jk in sorted(joint_keys):
        prefix = jk[: -len("hand_joints")]  # e.g. "right_"
        idx_key = f"{prefix}hand_joints_idx"
        j_shape = key_info[jk]["shape"]
        entry: dict[str, Any] = {"joints_shape": j_shape}
        if idx_key not in key_info:
            entry["idx_present"] = False
            report.add(
                f"hand_joints::{jk}", FAIL,
                f"'{jk}' present but '{idx_key}' missing",
            )
        else:
            entry["idx_present"] = True
            entry["idx_shape"] = key_info[idx_key]["shape"]
            # joints [T, N] vs idx [T, N] -> N must match.
            n_j = j_shape[1] if len(j_shape) >= 2 else 1
            n_i = key_info[idx_key]["shape"][1] if len(key_info[idx_key]["shape"]) >= 2 else 1
            entry["num_joints"] = n_j
            if n_j != n_i:
                report.add(
                    f"hand_joints::{jk}", FAIL,
                    f"joint dim mismatch: {jk} N={n_j} vs {idx_key} N={n_i}",
                )
            else:
                report.add(
                    f"hand_joints::{jk}", PASS,
                    f"'{jk}' paired with '{idx_key}' (N={n_j})",
                )
        summary[jk] = entry
    return summary


def _instruction_samples(data: Any, data_keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("sub_task_instruction", "high_level_task_instruction"):
        if k in data_keys:
            try:
                vals = np.asarray(data[k][:]).astype(str)
                uniq = sorted(set(v.strip() for v in vals.tolist()))
                out[k] = uniq[:10]
                out[f"{k}_num_unique"] = len(uniq)
            except Exception:  # noqa: BLE001
                out[k] = None
    return out


# --------------------------------------------------------------------------- #
# Website-renderable metadata (sources/<name>/metadata.json schema).
# --------------------------------------------------------------------------- #

def _derive_type(summaries: list[dict[str, Any]]) -> tuple[str, bool]:
    """Best-effort embodiment ``type``.

    Returns ``(type, confident)``. A single-DoF hand => 'gripper' (confident);
    a multi-DoF hand => 'dexhand' (not confident, since 'human' cannot be told
    apart from the data alone).
    """
    max_joints = 0
    saw_hand = False
    for s in summaries:
        for entry in (s.get("hand_joints") or {}).values():
            saw_hand = True
            max_joints = max(max_joints, int(entry.get("num_joints") or 1))
    if not saw_hand:
        return "gripper", False
    if max_joints <= 1:
        return "gripper", True
    return "dexhand", False


def _derive_sensors(
    summaries: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Aggregate tactile sensor names into (image, array, state, unknown_types)."""
    buckets: dict[str, set[str]] = {"image": set(), "array": set(), "state": set()}
    unknown_types: set[str] = set()
    for s in summaries:
        for tset in s.get("tactile_sets") or []:
            names = tset.get("sensor") or []
            for ttype in tset.get("type") or []:
                kind = TACTILE_TYPE_TO_SENSOR_KIND.get(str(ttype))
                if kind is None:
                    unknown_types.add(str(ttype))
                    continue
                buckets[kind].update(str(n) for n in names)
    return (
        sorted(buckets["image"]),
        sorted(buckets["array"]),
        sorted(buckets["state"]),
        sorted(unknown_types),
    )


def build_renderable_metadata(
    summaries: list[dict[str, Any]],
    source_name: str,
    merge_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate the per-dataset summaries into one website-schema metadata.json.

    Returns ``(metadata, export_info)`` where ``export_info`` documents what was
    auto-filled and what a human still needs to review.
    """
    valid = [s for s in summaries if s.get("num_frames") is not None]
    task = len(valid)
    trajectory = sum(int(s.get("num_episodes") or 0) for s in valid)
    frame = sum(int(s.get("num_frames") or 0) for s in valid)
    img, arr, st, unknown_types = _derive_sensors(valid)
    dtype, type_confident = _derive_type(valid)

    existing: dict[str, Any] = {}
    if merge_path is not None:
        existing = json.loads(Path(merge_path).read_text(encoding="utf-8"))

    def keep_or(field: str, fallback: Any) -> Any:
        val = existing.get(field)
        return val if val not in (None, "", []) else fallback

    # Human-curated sensor split / type take precedence when merging.
    type_final = keep_or("type", dtype)
    sensor_image = keep_or("sensor_image", img)
    sensor_array = keep_or("sensor_array", arr)
    sensor_state = keep_or("sensor_state", st)

    metadata = {
        "name": keep_or("name", source_name),
        "url": keep_or("url", []),
        "version": keep_or("version", ""),
        "description": keep_or("description", ""),
        "type": type_final,
        "sensor_image": sensor_image,
        "sensor_array": sensor_array,
        "sensor_state": sensor_state,
        "task": task,
        "trajectory": trajectory,
        "frame": frame,
        "label": keep_or("label", []),
        "contact": keep_or("contact", []),
        "reference": keep_or("reference", {"paper": "", "website": "", "authors": ""}),
        "code_trigger": keep_or("code_trigger", ""),
    }

    # Build human-review notes.
    needs_review: list[str] = []
    if not type_confident and not (existing.get("type")):
        needs_review.append(
            f"type='{metadata['type']}' was inferred heuristically; the zarr data alone "
            "cannot reliably distinguish dexhand from human. Please confirm."
        )
    if not existing:
        needs_review.append(
            "sensor_array/sensor_state assignment was inferred heuristically from tactile "
            "types (image->image, binary->array, state->state). Please confirm."
        )
    if unknown_types:
        needs_review.append(
            f"Unknown tactile type(s) found: {unknown_types} "
            "(not assigned to any sensor_* bucket)."
        )
    missing_human = [
        f for f in ("url", "version", "description", "contact", "code_trigger", "label")
        if not metadata.get(f)
    ]
    ref = metadata.get("reference") or {}
    if not any((ref.get("paper"), ref.get("website"), ref.get("authors"))):
        missing_human.append("reference")
    if missing_human:
        needs_review.append(f"Please fill these human-authored fields manually: {missing_human}")

    # Note any discrepancy between the derived sensors and a merged file.
    discrepancy: dict[str, Any] = {}
    if existing:
        derived = {"sensor_image": img, "sensor_array": arr, "sensor_state": st, "type": dtype}
        for field, dval in derived.items():
            eval_ = existing.get(field)
            if field == "type":
                if eval_ and dval and eval_ != dval:
                    discrepancy[field] = {"existing": eval_, "derived": dval}
            else:
                if sorted(eval_ or []) != sorted(dval or []):
                    discrepancy[field] = {"existing": eval_ or [], "derived": dval}
        if discrepancy:
            needs_review.append(
                f"Merged metadata differs from zarr-derived values for: {sorted(discrepancy)} "
                "(the merged-file values were kept)."
            )

    export_info = {
        "output": "metadata.json",
        "source_name": metadata["name"],
        "num_zarr_aggregated": task,
        "merged_from": str(merge_path) if merge_path else None,
        "auto_filled_from_zarr": [
            "task", "trajectory", "frame",
            "type", "sensor_image", "sensor_array", "sensor_state",
        ],
        "derived_counts": {"task": task, "trajectory": trajectory, "frame": frame},
        "needs_review": needs_review,
        "merge_discrepancy": discrepancy or None,
    }
    return metadata, export_info


# --------------------------------------------------------------------------- #
# Output / CLI.
# --------------------------------------------------------------------------- #

def _print_dataset_summary(report: DatasetReport) -> None:
    icon = "PASS" if report.passed else "FAIL"
    print(f"[{icon}] {report.name}  ({report.num_failed} error(s), {report.num_warn} warning(s))")
    for c in report.checks:
        if c.status in (FAIL, WARN):
            marker = "x" if c.status == FAIL else "!"
            print(f"    {marker} [{c.status}] {c.name}: {c.message}")


def _default_source_name(input_path: Path) -> str:
    name = input_path.expanduser().resolve().name
    if name.endswith(".zarr"):
        name = name[: -len(".zarr")]
    return name


def build_precheck_report_result_doc(
    header: dict[str, Any],
    reports: list[DatasetReport],
    export_info: dict[str, Any],
) -> dict[str, Any]:
    dataset_results: list[dict[str, Any]] = []
    failed_datasets: list[str] = []
    warning_datasets: list[str] = []
    needs_human_check = list(export_info.get("needs_review", []))

    for report in reports:
        failed_checks = [
            {"name": c.name, "message": c.message}
            for c in report.checks
            if c.status == FAIL
        ]
        warning_checks = [
            {"name": c.name, "message": c.message}
            for c in report.checks
            if c.status == WARN
        ]

        if failed_checks:
            final_status = "fail"
            failed_datasets.append(report.name)
        elif warning_checks:
            final_status = "warning"
            warning_datasets.append(report.name)
        else:
            final_status = "pass"

        dataset_results.append({
            "dataset": report.name,
            "final_status": final_status,
            "failed_checks": failed_checks,
            "warning_checks": warning_checks,
        })

    overall_status = "fail" if failed_datasets else ("warning" if warning_datasets or needs_human_check else "pass")

    return {
        "tool": header.get("tool"),
        "generated_at": header.get("generated_at"),
        "input_path": header.get("input_path"),
        "overall_status": overall_status,
        "failed_datasets": failed_datasets,
        "warning_datasets": warning_datasets,
        "needs_human_check": needs_human_check,
        "datasets": dataset_results,
    }


def run(
    input_path: Path,
    output_dir: Path,
    pattern: str,
    gap_factor: float,
    source_name: str | None = None,
    merge_metadata: Path | None = None,
    emit_metadata: bool = True,
) -> int:
    datasets = discover_datasets(input_path, pattern)
    print(f"Discovered {len(datasets)} dataset(s) under {input_path}\n")

    reports: list[DatasetReport] = []
    summaries: list[dict[str, Any]] = []
    for ds in datasets:
        try:
            report, summary = check_dataset(ds, gap_factor=gap_factor)
        except Exception as exc:  # noqa: BLE001 - never let one dataset kill the run
            report = DatasetReport(name=ds.name, path=str(ds))
            report.add("unexpected_error", FAIL, f"{type(exc).__name__}: {exc}")
            summary = {"name": ds.name, "path": str(ds), "error": str(exc)}
            traceback.print_exc()
        reports.append(report)
        summaries.append(summary)
        _print_dataset_summary(report)

    num_passed = sum(1 for r in reports if r.passed)
    num_failed = len(reports) - num_passed
    generated_at = _dt.datetime.now().isoformat(timespec="seconds")

    header = {
        "tool": "openxtactile_check_zarr",
        "version": __version__,
        "generated_at": generated_at,
        "input_path": str(input_path.expanduser().resolve()),
        "num_datasets": len(reports),
        "num_passed": num_passed,
        "num_failed": num_failed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- renderable metadata.json (aggregate all datasets as one source) ----- #
    metadata_path: Path | None = None
    dict_path: Path | None = None
    export_info: dict[str, Any] = {}
    if emit_metadata:
        name = source_name or _default_source_name(input_path)
        metadata, export_info = build_renderable_metadata(summaries, name, merge_metadata)
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metadata), f, ensure_ascii=False, indent=2)

    # --- dict.json: all keys + constant-like key summaries ------------------ #
    dict_doc: dict[str, Any]
    if len(summaries) == 1:
        dict_doc = _to_jsonable(summaries[0].get("dict_export") or {})
    else:
        dict_doc = {
            s.get("name", f"dataset_{idx}"): _to_jsonable(s.get("dict_export") or {})
            for idx, s in enumerate(summaries)
        }
    dict_path = output_dir / "dict.json"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(_format_json_compact_lists(dict_doc))
        f.write("\n")

    # --- precheck_report.json: checks + per-dataset technical metadata ------- #
    precheck_doc = {
        **header,
        "metadata_export": export_info or None,
        "datasets": [
            {**r.to_dict(), "metadata": _to_jsonable(s)}
            for r, s in zip(reports, summaries)
        ],
    }
    precheck_path = output_dir / "precheck_report.json"
    with open(precheck_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(precheck_doc), f, ensure_ascii=False, indent=2)

    precheck_result_doc = build_precheck_report_result_doc(header, reports, export_info or {})
    precheck_result_path = output_dir / "precheck_report_result.json"
    with open(precheck_result_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(precheck_result_doc), f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print(f"Datasets: {len(reports)} | passed: {num_passed} | failed: {num_failed}")
    print(f"Wrote check summary  -> {precheck_path}")
    print(f"Wrote compact result -> {precheck_result_path}")
    print(f"Wrote dict export    -> {dict_path}")
    if metadata_path is not None:
        counts = export_info.get("derived_counts", {})
        print(
            f"Wrote metadata.json  -> {metadata_path} "
            f"(task={counts.get('task')}, trajectory={counts.get('trajectory')}, "
            f"frame={counts.get('frame')})"
        )
        for note in export_info.get("needs_review", []):
            print(f"    review: {note}")

    return 0 if num_failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OXT zarr dataset precheck & metadata export tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A single '<name>.zarr' dataset OR a directory containing several.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.cwd() / "oxt_check_output",
        help="Directory for precheck_report.json / metadata_summary.json "
             "(default: ./oxt_check_output).",
    )
    parser.add_argument(
        "--pattern",
        default="*.zarr",
        help="Glob used to find datasets when 'path' is a directory "
             "(default: '*.zarr').",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=DEFAULT_GAP_FACTOR,
        help="A timestamp step larger than this multiple of the median step "
             "is flagged as a gap (default: 3.0).",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Name for the exported metadata.json (default: the input folder "
             "name). All *.zarr under 'path' are aggregated into this one source.",
    )
    parser.add_argument(
        "--merge-metadata",
        type=Path,
        default=None,
        help="Path to an existing metadata.json whose human-authored fields "
             "(url/description/contact/reference/...) are preserved while the "
             "counts are refreshed from the zarr.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip writing the renderable metadata.json (only run the checks).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(
            args.path,
            args.output_dir,
            args.pattern,
            args.gap_factor,
            source_name=args.source_name,
            merge_metadata=args.merge_metadata,
            emit_metadata=not args.no_metadata,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# python precheck_metadata.py FTP1_data_example\\MotionTrans\\robot_se+put_bread_to_the_red_pad+.zarr
