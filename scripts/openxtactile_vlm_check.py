#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image, ImageDraw

try:
    import zarr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires zarr. Install with:\n"
        "  pip install zarr numpy pillow\n"
        f"(import error: {exc})"
    )

try:
    from volcenginesdkarkruntime import Ark
except ImportError:  # pragma: no cover
    Ark = None

try:
    from openai import AzureOpenAI
except ImportError:  # pragma: no cover
    AzureOpenAI = None


TACTILE_TYPE_ALIAS = {
    "state": "state",
    "array": "array",
    "binary": "array",
    "matrix": "array",
    "image": "image",
}


@dataclass
class DatasetInput:
    path: Path
    name: str


def discover_datasets(path: Path, pattern: str = "*.zarr") -> list[DatasetInput]:
    p = path.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"path not found: {p}")
    if p.is_dir() and p.suffix == ".zarr":
        return [DatasetInput(path=p, name=p.name)]
    if p.is_dir():
        found = sorted(x for x in p.rglob(pattern) if x.is_dir())
        if found:
            return [DatasetInput(path=x, name=x.name) for x in found]
    raise ValueError("input must be a .zarr directory or a parent directory containing *.zarr")


def _as_np(arr: Any) -> np.ndarray:
    return np.asarray(arr)


def _normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.nanpercentile(x[finite], 5))
    hi = float(np.nanpercentile(x[finite], 95))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def tactile_frame_to_map(frame: np.ndarray) -> np.ndarray:
    # state: [C] or [C,1], array/image: [C,H,W] etc.
    if frame.ndim == 0:
        return frame.reshape(1, 1)
    if frame.ndim == 1:
        return frame.reshape(1, -1)
    if frame.ndim == 2:
        return frame
    c = frame.shape[0]
    return frame.reshape(c, -1)


def _resize_nn(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img
    ys = np.linspace(0, in_h - 1, out_h).astype(int)
    xs = np.linspace(0, in_w - 1, out_w).astype(int)
    return img[np.ix_(ys, xs)]


def _estimate_value_range(tactile_arr: Any, indices: list[int]) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    for i in indices[: min(len(indices), 24)]:
        f = np.asarray(tactile_arr[i], dtype=np.float32)
        if np.isfinite(f).any():
            vals.append(f.reshape(-1))
    if not vals:
        return 0.0, 1.0
    merged = np.concatenate(vals)
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, 5))
    hi = float(np.percentile(finite, 95))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _state_panel(
    frame: np.ndarray,
    energy_series: np.ndarray,
    t: int,
    out_size: tuple[int, int],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    w, h = out_size
    canvas = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    vec = frame.reshape(-1).astype(np.float32)
    if vec.size:
        bar_h = max(6, int(h * 0.45 / max(1, vec.size)))
        y0 = 8
        for i, v in enumerate(vec[: min(vec.size, 24)]):
            norm = 0.0 if vmax <= vmin else float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))
            bw = int(norm * (w - 24))
            yy = y0 + i * bar_h
            draw.rectangle([8, yy, 8 + bw, yy + bar_h - 2], fill=(120, 220, 120))
            draw.text((10, yy), f"c{i}", fill=(220, 220, 220))

    # mini sparkline of tactile energy
    win = 80
    s = max(0, t - win)
    e = min(len(energy_series), t + win + 1)
    seg = energy_series[s:e]
    if seg.size >= 2:
        seg_min = float(np.min(seg))
        seg_max = float(np.max(seg))
        if seg_max <= seg_min:
            seg_max = seg_min + 1e-6
        x0, y0, x1, y1 = 8, int(h * 0.58), w - 8, h - 10
        draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80))
        pts = []
        for i, val in enumerate(seg):
            xx = x0 + int(i * (x1 - x0) / max(1, len(seg) - 1))
            yy = y1 - int((float(val) - seg_min) / (seg_max - seg_min) * (y1 - y0 - 2))
            pts.append((xx, yy))
        if len(pts) >= 2:
            draw.line(pts, fill=(255, 200, 80), width=2)
        if s <= t < e:
            tx = x0 + int((t - s) * (x1 - x0) / max(1, len(seg) - 1))
            draw.line([(tx, y0), (tx, y1)], fill=(255, 80, 80), width=1)
    return np.asarray(canvas)


def _array_image_panel(
    frame: np.ndarray,
    out_size: tuple[int, int],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    w, h = out_size
    x = frame.astype(np.float32)
    if x.ndim == 2:
        maps = [x]
    elif x.ndim == 3:
        maps = [x[i] for i in range(x.shape[0])]
    else:
        # fallback: keep channel axis, preserve remaining spatial product
        c = x.shape[0] if x.ndim >= 1 else 1
        maps = [x[i].reshape(1, -1) for i in range(c)]

    n = len(maps)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = max(1, w // cols)
    cell_h = max(1, h // rows)
    panel = np.zeros((h, w), dtype=np.uint8)
    for idx, m in enumerate(maps):
        r, c = divmod(idx, cols)
        y0, x0 = r * cell_h, c * cell_w
        y1, x1 = min(h, y0 + cell_h), min(w, x0 + cell_w)
        if y1 <= y0 or x1 <= x0:
            continue
        mn = m
        norm = np.zeros_like(mn, dtype=np.float32) if vmax <= vmin else np.clip((mn - vmin) / (vmax - vmin), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        resized = _resize_nn(gray, y1 - y0, x1 - x0)
        panel[y0:y1, x0:x1] = resized
    return np.stack([panel, panel, panel], axis=2)


def render_tactile_panel(
    frame: np.ndarray,
    tactile_kind: str,
    energy_series: np.ndarray,
    t: int,
    out_size: tuple[int, int],
    vmin: float,
    vmax: float,
) -> np.ndarray:
    if tactile_kind == "state":
        return _state_panel(frame, energy_series, t, out_size, vmin, vmax)
    return _array_image_panel(frame, out_size, vmin, vmax)


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img
    f = img.astype(np.float32)
    if np.nanmax(f) <= 1.0:
        f = f * 255.0
    return np.clip(f, 0, 255).astype(np.uint8)


def build_sampling_indices(
    T: int,
    tactile_energy: np.ndarray,
    max_uniform: int,
    max_events: int,
    max_total: int,
) -> dict[str, Any]:
    if T <= 0:
        return {"indices": [], "uniform": [], "events": []}

    uniform = np.linspace(0, T - 1, min(max_uniform, T), dtype=int)
    diffs = np.abs(np.diff(tactile_energy.astype(np.float64), prepend=tactile_energy[0]))
    topk = min(max_events, T)
    events = np.argsort(diffs)[-topk:] if topk > 0 else np.array([], dtype=int)
    merged = sorted(set(uniform.tolist() + events.tolist()))

    if len(merged) > max_total:
        scores = {int(i): float(diffs[i]) for i in merged}
        base = uniform.tolist()
        rest = [x for x in merged if x not in set(base)]
        rest = sorted(rest, key=lambda x: scores.get(x, 0.0), reverse=True)
        merged = sorted(set(base + rest[: max(0, max_total - len(base))]))
        merged = merged[:max_total]

    return {
        "indices": merged,
        "uniform": uniform.tolist(),
        "events": sorted(events.tolist()),
    }


def make_montage(
    rgb_arr: Any,
    tactile_arr: Any,
    indices: list[int],
    tactile_kind: str,
    energy_series: np.ndarray,
    tactile_vmin: float,
    tactile_vmax: float,
    out_png: Path,
    tile_w: int = 224,
    tile_h: int = 224,
    cols: int = 4,
) -> None:
    if not indices:
        raise ValueError("empty indices for montage")

    rows = int(np.ceil(len(indices) / cols))
    panel_w = tile_w * 2
    canvas = Image.new("RGB", (panel_w * cols, tile_h * rows), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for n, t in enumerate(indices):
        r, c = divmod(n, cols)
        x0 = c * panel_w
        y0 = r * tile_h

        rgb = _to_uint8_rgb(_as_np(rgb_arr[t]))
        rgb_img = Image.fromarray(rgb).resize((tile_w, tile_h))
        canvas.paste(rgb_img, (x0, y0))

        tact = _as_np(tactile_arr[t]).astype(np.float32)
        panel = render_tactile_panel(
            frame=tact,
            tactile_kind=tactile_kind,
            energy_series=energy_series,
            t=t,
            out_size=(tile_w, tile_h),
            vmin=tactile_vmin,
            vmax=tactile_vmax,
        )
        heat_img = Image.fromarray(panel).resize((tile_w, tile_h))
        canvas.paste(heat_img, (x0 + tile_w, y0))

        draw.text((x0 + 6, y0 + 6), f"t={t}", fill=(255, 255, 0))
        draw.text((x0 + tile_w + 6, y0 + 6), f"type={tactile_kind}", fill=(255, 255, 255))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)


def encode_image_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _response_to_dict(resp: Any) -> dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        try:
            return resp.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(resp, "to_dict"):
        try:
            return resp.to_dict()
        except Exception:  # noqa: BLE001
            pass
    try:
        return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:  # noqa: BLE001
        return {"raw": str(resp)}


def call_openai_compatible_chat(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    timeout_s: int = 60,
) -> dict[str, Any]:
    endpoint = api_base.rstrip("/") + "/chat/completions"

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_path is not None:
        b64 = encode_image_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="ignore")
        return {"_error": f"HTTP {exc.code}", "_body": err_text}
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def call_ark_responses(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
) -> dict[str, Any]:
    endpoint = api_base.rstrip("/") + "/responses"
    if image_path is None:
        input_payload: Any = prompt
    else:
        b64 = encode_image_base64(image_path)
        input_payload = [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                {"type": "input_text", "text": prompt},
            ],
        }]
    payload = {
        "model": model,
        "input": input_payload,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="ignore")
        return {"_error": f"HTTP {exc.code}", "_body": err_text}
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def call_vlm(
    provider: str,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    azure_api_version: str = "2025-03-01-preview",
) -> dict[str, Any]:
    if provider == "azure":
        if AzureOpenAI is None:
            return {"_error": "missing_dependency: openai is not installed"}
        try:
            client = AzureOpenAI(
                azure_endpoint=api_base,
                api_key=api_key,
                api_version=azure_api_version,
            )
            if image_path is None:
                input_payload: Any = prompt
            else:
                b64 = encode_image_base64(image_path)
                input_payload = [{
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"},
                        {"type": "input_text", "text": prompt},
                    ],
                }]
            resp = client.responses.create(
                model=model,
                input=input_payload,
            )
            return _response_to_dict(resp)
        except Exception as exc:  # noqa: BLE001
            return {"_error": str(exc)}
    if provider == "ark":
        return call_ark_responses(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_path=image_path,
        )
    return call_openai_compatible_chat(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_path=image_path,
    )


def extract_model_text(resp: dict[str, Any]) -> str:
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    output = resp.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for c in item.get("content", []) or []:
                if isinstance(c, dict):
                    t = c.get("text")
                    if isinstance(t, str):
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return ""


def robust_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            return {"raw_text": text}
    return {"raw_text": text}


def summarize_tactile_health(tactile_energy: np.ndarray) -> dict[str, Any]:
    if tactile_energy.size == 0:
        return {"ok": False, "reason": "empty_tactile_energy"}
    diff = np.abs(np.diff(tactile_energy))
    dead_ratio = float(np.mean(diff < 1e-7)) if diff.size else 1.0
    p95 = float(np.percentile(tactile_energy, 95))
    p5 = float(np.percentile(tactile_energy, 5))
    dynamic_range = p95 - p5
    return {
        "ok": True,
        "dead_step_ratio": dead_ratio,
        "dynamic_range_p95_p5": dynamic_range,
        "min": float(np.min(tactile_energy)),
        "max": float(np.max(tactile_energy)),
        "mean": float(np.mean(tactile_energy)),
    }


def analyze_one_dataset(
    ds: DatasetInput,
    out_dir: Path,
    max_uniform_samples: int,
    max_event_samples: int,
    max_frames_for_vlm: int,
    provider: str,
    api_base: str,
    api_key: str | None,
    model: str,
    azure_api_version: str,
    dry_run: bool,
) -> dict[str, Any]:
    root = zarr.open_group(str(ds.path), mode="r")
    data = root["data"] if "data" in root else root
    keys = sorted(data.array_keys())
    rgb_keys = [k for k in keys if k.endswith("_rgb")]
    tactile_data_keys = [k for k in keys if "_tactile_data_" in k]
    instruction_keys = [k for k in keys if "instruction" in k]

    if not rgb_keys or not tactile_data_keys:
        return {
            "dataset": ds.name,
            "path": str(ds.path),
            "status": "fail",
            "reason": "missing_rgb_or_tactile_data",
            "rgb_keys": rgb_keys,
            "tactile_data_keys": tactile_data_keys,
        }

    rgb_key = rgb_keys[0]
    tact_key = tactile_data_keys[0]
    rgb = data[rgb_key]
    tact = data[tact_key]
    T = int(min(rgb.shape[0], tact.shape[0]))

    tactile_energy = np.asarray([float(np.nanmean(np.abs(np.asarray(tact[i], dtype=np.float32)))) for i in range(T)])
    sampling = build_sampling_indices(
        T,
        tactile_energy,
        max_uniform=max_uniform_samples,
        max_events=max_event_samples,
        max_total=max_frames_for_vlm,
    )
    indices = sampling["indices"]

    tactile_type_values: list[str] = []
    type_key = tact_key.replace("_tactile_data_", "_tactile_type_")
    if type_key in keys:
        vals = np.asarray(data[type_key][:]).astype(str)
        tactile_type_values = sorted(set(vals.tolist()))
    tactile_type_norm = sorted({TACTILE_TYPE_ALIAS.get(v, v) for v in tactile_type_values})
    tactile_kind = tactile_type_norm[0] if tactile_type_norm else "array"
    tactile_vmin, tactile_vmax = _estimate_value_range(tact, indices)

    dataset_out = out_dir / ds.name.replace(".zarr", "")
    dataset_out.mkdir(parents=True, exist_ok=True)
    montage_png = dataset_out / "vlm_montage.png"
    make_montage(
        rgb_arr=rgb,
        tactile_arr=tact,
        indices=indices,
        tactile_kind=tactile_kind,
        energy_series=tactile_energy,
        tactile_vmin=tactile_vmin,
        tactile_vmax=tactile_vmax,
        out_png=montage_png,
    )

    instructions: list[str] = []
    for ik in instruction_keys:
        try:
            vals = np.asarray(data[ik][:]).astype(str)
            uniq = sorted(set(x.strip() for x in vals.tolist()))
            instructions.extend(uniq[:3])
        except Exception:  # noqa: BLE001
            continue
    instructions = sorted(set(instructions))[:5]

    heuristic = summarize_tactile_health(tactile_energy)
    prompt = (
        "You are reviewing aligned RGB+tactile snapshots from a robotics dataset.\n"
        "Return STRICT JSON only with keys:\n"
        "sync_score(0-1), contact_consistency_score(0-1), signal_health_score(0-1), "
        "instruction_consistency_score(0-1), issues(array of short strings), "
        "final_status(one of: pass, warning, fail), needs_human_review(boolean).\n"
        f"Known tactile type labels: {tactile_type_values}.\n"
        f"Normalized tactile type labels: {tactile_type_norm}.\n"
        f"Instruction samples: {instructions}.\n"
        "Tactile panel rendering rule:\n"
        "- if type=state: top half are per-channel bars, bottom half is local energy curve.\n"
        "- if type=array/image: channel maps are tiled while preserving channel 2D structure.\n"
        f"- fixed tactile normalization range for this dataset: [{tactile_vmin:.6g}, {tactile_vmax:.6g}].\n"
        "Focus on: RGB-tactile rough synchronization, broken contact patterns, abnormal "
        "signal stability, and instruction/action consistency.\n"
    )

    vlm_resp: dict[str, Any] = {}
    vlm_json: dict[str, Any] = {}
    if dry_run:
        vlm_json = {
            "sync_score": None,
            "contact_consistency_score": None,
            "signal_health_score": None,
            "instruction_consistency_score": None,
            "issues": ["dry_run_no_api_call"],
            "final_status": "warning",
            "needs_human_review": True,
        }
    elif api_key:
        vlm_resp = call_vlm(
            provider=provider,
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            image_path=montage_png,
            azure_api_version=azure_api_version,
        )
        text = extract_model_text(vlm_resp)
        vlm_json = robust_json_from_text(text)
        if "_error" in vlm_resp and not vlm_json:
            vlm_json = {
                "issues": [f"api_error: {vlm_resp.get('_error')}"],
                "final_status": "warning",
                "needs_human_review": True,
            }
    else:
        vlm_json = {
            "issues": ["missing_api_key"],
            "final_status": "warning",
            "needs_human_review": True,
        }

    return {
        "dataset": ds.name,
        "path": str(ds.path),
        "status": "ok",
        "selected_rgb_key": rgb_key,
        "selected_tactile_key": tact_key,
        "num_frames": T,
        "sampling": {
            "strategy": "uniform + top_tactile_change",
            "max_uniform_samples": max_uniform_samples,
            "max_event_samples": max_event_samples,
            "max_frames_for_vlm": max_frames_for_vlm,
            **sampling,
        },
        "tactile_type_raw": tactile_type_values,
        "tactile_type_normalized": tactile_type_norm,
        "tactile_render_kind": tactile_kind,
        "tactile_render_range": {"vmin": tactile_vmin, "vmax": tactile_vmax},
        "instruction_samples": instructions,
        "heuristic_signal_summary": heuristic,
        "artifacts": {
            "montage_png": str(montage_png),
        },
        "vlm_review": vlm_json,
        "vlm_raw_response": vlm_resp if vlm_resp else None,
    }


def test_api_connectivity(
    provider: str,
    api_base: str,
    api_key: str,
    model: str,
    azure_api_version: str,
) -> dict[str, Any]:
    prompt = "Reply exactly with JSON: {\"pong\": true}"
    resp = call_vlm(
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=prompt,
        image_path=None,
        azure_api_version=azure_api_version,
    )
    out = {"ok": False, "provider": provider, "api_base": api_base, "model": model}
    if "_error" in resp:
        out["error"] = resp["_error"]
        out["body"] = resp.get("_body")
        return out
    txt = extract_model_text(resp)
    out["raw_text"] = txt
    parsed = robust_json_from_text(txt)
    out["parsed"] = parsed
    out["ok"] = bool(parsed.get("pong") is True)
    return out


def run(args: argparse.Namespace) -> int:
    api_key = args.api_key
    if not api_key:
        if args.provider == "ark":
            env_name = "ARK_API_KEY"
        elif args.provider == "azure":
            env_name = "AZURE_API_KEY"
        else:
            env_name = "VLM_API_KEY"
        api_key = os.environ.get(env_name)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.test_api:
        if not api_key:
            print("error: api key is required for --test-api", file=sys.stderr)
            return 2
        result = test_api_connectivity(
            args.provider,
            args.api_base,
            api_key,
            args.model,
            args.azure_api_version,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("ok") else 1

    datasets = discover_datasets(args.path, args.pattern)
    reports: list[dict[str, Any]] = []
    for ds in datasets:
        report = analyze_one_dataset(
            ds=ds,
            out_dir=output_dir,
            max_uniform_samples=args.max_uniform_samples,
            max_event_samples=args.max_event_samples,
            max_frames_for_vlm=args.max_frames_for_vlm,
            provider=args.provider,
            api_base=args.api_base,
            api_key=api_key,
            model=args.model,
            azure_api_version=args.azure_api_version,
            dry_run=args.dry_run,
        )
        reports.append(report)
        status = report.get("status")
        print(f"[{status}] {ds.name}")

    out = {
        "tool": "openxtactile_vlm_check",
        "provider": args.provider,
        "api_base": args.api_base,
        "model": args.model,
        "azure_api_version": args.azure_api_version if args.provider == "azure" else None,
        "num_datasets": len(reports),
        "reports": reports,
    }
    out_path = output_dir / "vlm_review.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VLM-based semantic check for OXT zarr datasets with aggressive downsampling."
    )
    p.add_argument("path", type=Path, help="One *.zarr path or a directory containing *.zarr")
    p.add_argument("-o", "--output-dir", type=Path, default=Path.cwd() / "vlm_check_output")
    p.add_argument("--pattern", default="*.zarr")
    p.add_argument("--provider", choices=["ark", "openai", "azure"], default="ark")
    p.add_argument("--api-base", default="https://ark.cn-beijing.volces.com/api/v3")
    p.add_argument("--model", default="doubao-seed-2-1-pro-260628")
    p.add_argument(
        "--azure-api-version",
        default="2025-03-01-preview",
        help="Used only when provider=azure (Responses API requires 2025-03-01-preview or later).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to ARK_API_KEY (ark), AZURE_API_KEY (azure), or VLM_API_KEY (openai).",
    )
    p.add_argument("--max-uniform-samples", type=int, default=12)
    p.add_argument("--max-event-samples", type=int, default=12)
    p.add_argument("--max-frames-for-vlm", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="Skip API calls; still produce montage/output")
    p.add_argument("--test-api", action="store_true", help="Only test API connectivity and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
