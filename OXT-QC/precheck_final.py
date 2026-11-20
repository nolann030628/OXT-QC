#!/usr/bin/env python3
"""Final OXT submission review: 14 main checks plus 10.1/10.2/10.3.

Input is a final metadata directory, NOT a raw Zarr. Reports are written beside
metadata.json. --batch checks immediate submission folders and writes a summary.
Reports are in English, with an additional Chinese Markdown report.
No data, spreadsheet, remote repository, or existing QC result is modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


VERSION = "2.0.0"
RULES_CN = {
    "01": "image / RGB dtype 与 shape",
    "02": "state / matrix 双向类型、shape 与触觉成员",
    "03": "操作状态、关节索引与功能区域",
    "04": "ego / umi 的 Viser 4D 录像",
    "05": "标准 pose 维度",
    "06": "func_area、sensor、hand joint、institution 图片",
    "07": "supplementary 文件、实际数据与 with-supp",
    "08": "示例视频格式",
    "09": "指令字段、dtype 与 shape",
    "10": "metadata、英文、邮箱、标签及统计",
    "10.1": "_OXT 后缀和仓库内目录名称一致性",
    "10.2": "仓库同名目录的两种压缩包格式",
    "10.3": "HF / hf-mirror 访问和覆盖",
    "11": "QC 文件、VLM 执行与评分",
    "12": "传感器名称和图片对应",
    "13": "supplementary 说明结构与维度",
    "14": "ego 相机字段",
}
RULES = {
    "01": "Image / RGB dtype and shape",
    "02": "State / matrix types, shapes, and tactile members",
    "03": "Action states, joint indices, and functional areas",
    "04": "Viser 4D recording for ego / umi",
    "05": "Standard pose dimensions",
    "06": "Functional-area, sensor, hand-joint, and institution images",
    "07": "Supplementary files, data, and with-supp label",
    "08": "Example video format",
    "09": "Instruction fields, dtype, and shape",
    "10": "Metadata fields, English, email, labels, and statistics",
    "10.1": "_OXT suffix and repository directory-name consistency",
    "10.2": "The two archive layouts under the repository-named directory",
    "10.3": "HF / hf-mirror access and coverage",
    "11": "QC files, VLM execution, and scores",
    "12": "Sensor names and image correspondence",
    "13": "Supplementary note structure and dimensions",
    "14": "Ego camera fields",
}
ORDER = {key: i for i, key in enumerate(RULES)}
STATUS_ORDER = {"not_applicable": 0, "pass": 1, "unknown": 2, "warning": 3, "fail": 4}
TACTILE_RE = re.compile(r"^(.+)_tactile_(data|area|sensor|type)_(.+)$")
STATE_RE = re.compile(r"^(left|right)_(arm_joints|wrist_pose|hand_joints|hand_pose|hand_keypoints)$")
POSE_RE = re.compile(r"^(left|right)_wrist_pose$|^camera_ego_pose$")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
EMAIL_RE = re.compile(r"^[^\s@<>,;]+@[^\s@<>,;]+\.[^\s@<>,;]+$")
UINT8 = {"uint8", "u1", "|u1", "<u1", ">u1", "=u1"}
IMAGE_DTYPE_GUIDANCE_CN = "image 数据不是 uint8：浮点等数值场请将 type 改为 matrix；如果确为 image，请将数据 dtype 改为 uint8"
IMAGE_DTYPE_GUIDANCE = "Image data is not uint8: use type=matrix for floating-point or other numeric fields; if it is truly an image, use dtype=uint8"
TYPES = {"state", "binary", "image", "matrix"}
LABELS = {"ego", "umi", "with-supp"}
FIELDS = ("name", "url", "version", "description", "type", "sensor_image", "sensor_array",
          "sensor_state", "task", "trajectory", "frame", "label", "contact", "reference", "code_trigger")
ASPECTS = ("sync", "contact_consistency", "signal_health", "instruction_consistency")
ARCHIVE_RE = re.compile(r"\.(?:zip|7z|rar|tar|tgz|tbz2?|txz|gz|bz2|xz|zst|zstd|lz4|lz|lzma|lzo|cab|arj|ace|cpio|iso|paq\d*|zpaq|br)(?:\.\d{2,})?$|\.(?:z\d{2,}|r\d{2,})$", re.I)
NON_ARCHIVE = {".json", ".jsonl", ".csv", ".md", ".txt", ".yaml", ".yml", ".py", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".pdf", ".html", ".sha256", ".npy", ".npz", ".parquet", ".h5", ".hdf5"}
# Explicit legacy identities are identification aids, never accepted spellings.


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def flatten(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return [part for item in value for part in flatten(item)]
    return [] if value is None else [value]


def strings(value: Any) -> list[str]:
    return [str(v) for v in flatten(value)]




def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_entries(node: Any) -> dict:
    if not isinstance(node, dict):
        return {}
    return {k: v for k, v in node.items() if isinstance(v, dict) and isinstance(v.get("shape"), list)}


def iter_tables(node: Any, path: tuple = ()):
    if not isinstance(node, dict):
        return
    # A QC wrapper can carry three copies of the same arrays. Use one export.
    for wrapper in ("dict_export", "keys"):
        entries = array_entries(node.get(wrapper))
        if entries:
            meta = node.get("meta_keys")
            if isinstance(meta, dict):
                entries = {**entries, **{"meta/" + k: v for k, v in array_entries(meta).items()}}
            yield path, entries
            return
    entries = array_entries(node)
    if entries:
        yield path, entries
        return
    for key, value in node.items():
        if isinstance(value, dict):
            yield from iter_tables(value, path + (key,))


def walk_strings(value: Any, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            if CJK_RE.search(str(key)):
                yield path + "." + str(key) + "[key]", str(key)
            yield from walk_strings(item, path + "." + str(key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk_strings(item, f"{path}[{i}]")
    elif isinstance(value, str):
        yield path.lstrip("."), value


def action_keys(table: dict) -> list[str]:
    result = []
    for key in table:
        lower = key.lower()
        if STATE_RE.match(key):
            result.append(key)
        elif not any(x in lower for x in ("tactile", "camera", "idx", "index", "timestamp", "valid", "metadata", "instruction", "error")):
            if re.search(r"joint|(?:^|_)action(?:s|_|$)|(?:^|_)pose(?:_|$)|hand_fingertip_positions", lower):
                result.append(key)
    return result


@dataclass
class Options:
    offline: bool = False
    timeout: float = 8.0
    repo_budget: float = 40.0
    max_pages: int = 12
    max_tasks: int = 100
    max_entries: int = 20000
    archive_samples: int = 1
    hf_endpoint: str = "https://huggingface.co"
    mirror_endpoint: str = "https://hf-mirror.com"
    ms_endpoint: str = "https://modelscope.cn"


class Collector:
    def __init__(self):
        self.findings: list[dict] = []
        self._groups: dict[tuple, dict] = {}
        self.applicable: set[str] = set()

    def add(self, rule: str, status: str, code: str, message: str, *, key: str = "", source: str = "",
            evidence: Any = None, example: str = "", signature: Any = None):
        message = message_text(code, message)
        self.applicable.add(rule)
        identity = (rule, status, code, key, source, str(signature) if signature is not None else message)
        if identity in self._groups:
            item = self._groups[identity]
            item["occurrences"] += 1
            if len(item["examples"]) < 3 and (example or evidence is not None):
                item["examples"].append({"table": example, "evidence": evidence})
            return item
        item = {"rule": rule, "status": status, "code": code, "message": message, "key": key,
                "source": source, "occurrences": 1, "examples": []}
        if example or evidence is not None:
            item["examples"].append({"table": example, "evidence": evidence})
        self.findings.append(item)
        self._groups[identity] = item
        return item

    def summary(self):
        result = []
        for rule, title in RULES.items():
            items = [f for f in self.findings if f["rule"] == rule]
            status = max((f["status"] for f in items), key=lambda s: STATUS_ORDER[s]) if items else ("pass" if rule in self.applicable else "not_applicable")
            result.append({"rule": rule, "title": title, "status": status, "finding_groups": len(items),
                           "status_counts": dict(Counter(f["status"] for f in items))})
        return result




@dataclass(frozen=True)
class RemoteSource:
    provider: str
    repo: str
    revision: str | None = None
    subdir: str = ""
    filename: str = ""
    mirror: bool = False


def parse_source(url: str) -> RemoteSource:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Dataset URLs must use HTTPS without embedded credentials")
    host = (parsed.hostname or "").lower().removeprefix("www.")
    provider = "hf" if host in ("huggingface.co", "hf-mirror.com") else "ms" if host == "modelscope.cn" else None
    parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/")]
    if provider is None or len(parts) < 3 or parts[0] != "datasets":
        raise ValueError("Not a supported HF/MS dataset URL")
    if any(p in ("", ".", "..") or "/" in p or "\\" in p for p in parts[1:3]):
        raise ValueError("Invalid repository owner/name")
    revision = None
    subdir = filename = ""
    if len(parts) > 3:
        if parts[3] == "files" and provider == "ms" and len(parts) == 4:
            pass
        elif parts[3] in ("tree", "blob", "resolve") and len(parts) >= 5:
            revision = parts[4]
            tail = "/".join(parts[5:])
            if parts[3] == "tree":
                subdir = tail
            else:
                filename = tail
        else:
            raise ValueError("The URL path format is ambiguous")
    query = urllib.parse.parse_qs(parsed.query)
    if provider == "ms":
        revision = query.get("version", query.get("Revision", [revision]))[0]
        subdir = query.get("path", [subdir])[0].strip("/")
    for value in (revision, subdir, filename):
        if value and ("\\" in value or any(p in (".", "..") for p in value.split("/"))):
            raise ValueError("The URL contains an unsafe path")
    return RemoteSource(provider, "/".join(parts[1:3]), revision, subdir, filename, host == "hf-mirror.com")


class RemoteError(Exception):
    pass


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        redirected = super().redirect_request(request, fp, code, message, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(request.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


@dataclass
class Tree:
    entries: list[dict] = field(default_factory=list)
    complete: bool = True
    pages: int = 0
    reason: str = ""


class RemoteClient:
    def __init__(self, options: Options):
        self.options = options
        self.opener = urllib.request.build_opener(SafeRedirect())
        self._cache: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        self._repo_locks: dict[tuple, threading.Lock] = {}

    def request(self, url: str, deadline: float, *, limit: int = 3_000_000, as_json: bool = True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RemoteError("The repository time budget was exhausted")
        headers = {"User-Agent": "OXT-precheck-final/" + VERSION, "Accept": "application/json" if as_json else "*/*"}
        # Explicit official-host tokens are not propagated to mirror endpoints.
        host = urllib.parse.urlsplit(url).hostname
        if host == "huggingface.co" and os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        if host in ("modelscope.cn", "www.modelscope.cn"):
            token = os.environ.get("MODELSCOPE_API_TOKEN") or os.environ.get("MODELSCOPE_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
        if not as_json:
            headers["Range"] = f"bytes=0-{limit - 1}"
        try:
            with self.opener.open(urllib.request.Request(url, headers=headers), timeout=min(self.options.timeout, remaining)) as response:
                body = response.read(limit + 1)
                if as_json and len(body) > limit:
                    raise RemoteError("The API response exceeded the bounded read limit; listing completeness is unknown")
                metadata = {"link": response.headers.get("Link"), "url": response.url}
                if as_json:
                    try:
                        return json.loads(body.decode("utf-8")), metadata
                    except (ValueError, UnicodeError) as error:
                        raise RemoteError("The API did not return parseable JSON") from error
                return body[:limit], metadata
        except urllib.error.HTTPError as error:
            raise RemoteError(f"HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
            raise RemoteError(f"Network access failed: {type(error).__name__}") from error

    @staticmethod
    def entries(raw: list, parent: str) -> list[dict]:
        result = []
        for item in raw:
            if not isinstance(item, dict):
                raise RemoteError("A directory entry is not an object")
            path = str(item.get("path", item.get("Path", ""))).replace("\\", "/").strip("/")
            if not path or any(p in (".", "..") for p in path.split("/")):
                raise RemoteError("A directory entry has a missing or invalid path")
            if parent and not path.startswith(parent + "/"):
                if "/" not in path:
                    path = parent + "/" + path
                else:
                    raise RemoteError("The API returned an entry outside the requested directory")
            tail = path[len(parent) + 1:] if parent else path
            if "/" in tail:
                raise RemoteError("A non-recursive listing returned deep entries; direct-child coverage is unclear")
            typ = item.get("type", item.get("Type"))
            kind = "directory" if typ in ("directory", "tree") else "file" if typ in ("file", "blob") else None
            if kind is None:
                raise RemoteError("Unknown directory entry type")
            result.append({"path": path, "kind": kind, "size": item.get("size", item.get("Size")),
                           "lfs": bool(item.get("lfs", item.get("IsLFS", False)))})
        return result

    def list_hf(self, endpoint: str, source: RemoteSource, revision: str, parent: str, deadline: float) -> Tree:
        url = f"{endpoint}/api/datasets/{urllib.parse.quote(source.repo, safe='/')}/tree/{urllib.parse.quote(revision, safe='')}"
        if parent:
            url += "/" + urllib.parse.quote(parent, safe="/")
        url += "?" + urllib.parse.urlencode({"recursive": "false", "expand": "false", "limit": 1000})
        result = Tree()
        seen = set()
        seen_urls = set()
        while url:
            if url in seen_urls:
                raise RemoteError("HF pagination loop detected")
            seen_urls.add(url)
            raw, headers = self.request(url, deadline)
            if not isinstance(raw, list):
                raise RemoteError("Invalid HF tree response format")
            page = self.entries(raw, parent)
            for entry in page:
                if entry["path"] in seen:
                    raise RemoteError("HF pagination repeated an entry; the snapshot may have changed")
                seen.add(entry["path"])
                result.entries.append(entry)
            result.pages += 1
            link = headers.get("link") or ""
            match = re.search(r'<([^>]+)>;\s*rel="?next"?', link)
            url = urllib.parse.urljoin(endpoint, match[1]) if match else ""
            if url and urllib.parse.urlsplit(url).netloc != urllib.parse.urlsplit(endpoint).netloc:
                raise RemoteError("The HF pagination hostname changed")
            if len(result.entries) > self.options.max_entries or (url and result.pages >= self.options.max_pages):
                result.complete = False
                result.reason = "The entry/page limit was reached"
                break
        return result

    def list_ms(self, source: RemoteSource, parent: str, deadline: float) -> Tree:
        endpoint = self.options.ms_endpoint
        result = Tree()
        seen = set()
        route = source.repo
        page_number = 1
        total = None
        while True:
            params = {"Revision": source.revision or "master", "Root": parent or "/", "Recursive": "false", "PageNumber": page_number, "PageSize": 100}
            url = f"{endpoint}/api/v1/datasets/{urllib.parse.quote(route, safe='/')}/repo/tree?{urllib.parse.urlencode(params)}"
            try:
                raw, _ = self.request(url, deadline)
            except RemoteError as error:
                if page_number != 1 or route != source.repo or str(error) not in ("HTTP 400", "HTTP 404"):
                    raise
                info, _ = self.request(f"{endpoint}/api/v1/datasets/{urllib.parse.quote(source.repo, safe='/')}", deadline)
                data = info.get("Data") if isinstance(info, dict) else None
                dataset_id = data.get("Id") if isinstance(data, dict) else None
                if dataset_id is None:
                    raise error
                route = str(dataset_id)
                continue
            if not isinstance(raw, dict) or raw.get("Code") not in (200, "200") or not isinstance(raw.get("Data"), dict):
                raise RemoteError("The MS tree API returned unsuccessful Data")
            raw_files = raw["Data"].get("Files")
            if not isinstance(raw_files, list):
                raise RemoteError("The MS tree response has no Files list")
            page = self.entries(raw_files, parent)
            for entry in page:
                if entry["path"] in seen:
                    raise RemoteError("MS pagination repeated an entry; completeness is unknown")
                seen.add(entry["path"])
                result.entries.append(entry)
            reported_total = raw.get("TotalCount", raw["Data"].get("TotalCount"))
            if isinstance(reported_total, int):
                if total is not None and reported_total != total:
                    raise RemoteError("MS pagination TotalCount changed")
                total = reported_total
            result.pages += 1
            if total is not None and len(result.entries) >= total:
                break
            if len(page) < 100:
                if total is not None and len(result.entries) < total:
                    result.complete = False
                    result.reason = "MS returned a short page before TotalCount was reached"
                break
            if len(result.entries) > self.options.max_entries or result.pages >= self.options.max_pages:
                result.complete = False
                result.reason = "The entry/page limit was reached"
                break
            page_number += 1
        return result

    def file(self, endpoint: str, source: RemoteSource, revision: str, path: str, deadline: float, limit: int = 65536):
        if source.provider == "hf":
            url = f"{endpoint}/datasets/{urllib.parse.quote(source.repo, safe='/')}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path, safe='/')}"
        else:
            url = f"{endpoint}/api/v1/datasets/{urllib.parse.quote(source.repo, safe='/')}/repo?" + urllib.parse.urlencode({"Revision": revision, "FilePath": path})
        return self.request(url, deadline, limit=limit, as_json=False)[0]

    def root(self, source: RemoteSource) -> dict:
        identity = (source.provider, source.repo, source.revision, source.mirror)
        with self._lock:
            lock = self._repo_locks.setdefault(identity, threading.Lock())
        with lock:
            if identity in self._cache:
                return self._cache[identity]
            deadline = time.monotonic() + self.options.repo_budget
            result = {"provider": source.provider, "repo": source.repo, "requested_revision": source.revision,
                      "revision": source.revision or ("main" if source.provider == "hf" else "master"),
                      "pinned": False, "endpoint": "", "attempts": [], "accessed_at": timestamp()}
            endpoints = [self.options.ms_endpoint] if source.provider == "ms" else ([self.options.mirror_endpoint] if source.mirror else [self.options.hf_endpoint, self.options.mirror_endpoint])
            for endpoint in endpoints:
                try:
                    revision = result.get("official_sha", result["revision"])
                    pinned = bool(re.fullmatch(r"[0-9a-fA-F]{40}", revision))
                    if source.provider == "hf":
                        try:
                            info_url = f"{endpoint}/api/datasets/{urllib.parse.quote(source.repo, safe='/')}"
                            info_revision = result.get("official_sha") or source.revision
                            if info_revision:
                                info_url += "/revision/" + urllib.parse.quote(info_revision, safe="")
                            info, _ = self.request(info_url, deadline)
                            sha = info.get("sha") if isinstance(info, dict) else None
                            if isinstance(sha, str) and re.fullmatch(r"[0-9a-fA-F]{40}", sha):
                                if result.get("official_sha") and sha != result["official_sha"]:
                                    raise RemoteError("The mirror revision differs from the known official commit")
                                revision = sha
                                pinned = True
                                if endpoint == self.options.hf_endpoint:
                                    result["official_sha"] = sha
                        except RemoteError as error:
                            result["attempts"].append({"endpoint": endpoint, "stage": "revision", "error": str(error)})
                        tree = self.list_hf(endpoint, source, revision, "", deadline)
                    else:
                        tree = self.list_ms(source, "", deadline)
                    result.update(endpoint=endpoint, revision=revision, pinned=pinned, root=asdict(tree))
                    result["attempts"].append({"endpoint": endpoint, "stage": "root", "status": "ok"})
                    break
                except RemoteError as error:
                    result["attempts"].append({"endpoint": endpoint, "stage": "root", "error": str(error)})
            if "root" not in result:
                result["error"] = "Neither the official endpoint nor the mirror completed the listing" if source.provider == "hf" else "The MS directory could not be read"
            self._cache[identity] = result
            return result


def archive_kind(path: str, prefix: bytes | None = None) -> str:
    """Known archive, confirmed plain file, or unknown; formats aren't restricted."""
    if prefix is not None:
        if prefix.startswith(b"version https://git-lfs.github.com/spec"):
            return "pointer"
        if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"7z\xbc\xaf\x27\x1c", b"Rar!", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd")) or prefix[257:262] == b"ustar":
            return "archive"
        # A zip filename plus an HTML/API error response is not proof of data.
        return "unknown"
    name = PurePosixPath(path).name
    if ARCHIVE_RE.search(name):
        return "archive"
    if name in (".DS_Store", ".gitkeep", ".gitattributes", "LICENSE") or PurePosixPath(name).suffix.lower() in NON_ARCHIVE:
        return "plain"
    return "unknown"


def check_layout(entries: list[dict], dataset_name: str, children: dict[str, Tree], collector: Collector, *, complete: bool = True):
    """Validate only contents INSIDE DatasetName; repository README is allowed."""
    collector.applicable.add("10.2")
    directories = [e for e in entries if e["kind"] == "directory"]
    files = [e for e in entries if e["kind"] == "file"]
    if not complete:
        collector.add("10.2", "unknown", "tree_partial", 'The dataset directory listing is incomplete', source=dataset_name)
    if not entries and complete:
        collector.add("10.2", "fail", "empty_dataset", 'The dataset directory is empty', source=dataset_name)
    if directories and files:
        collector.add("10.2", "fail", "mixed_layout", 'The dataset directory mixes files and task directories', source=dataset_name,
                      evidence={"files": [e["path"] for e in files[:10]], "directories": [e["path"] for e in directories[:10]]})
    levels = [(dataset_name, files)] if not directories else []
    for directory in directories:
        child = children.get(directory["path"])
        if child is None:
            collector.add("10.2", "unknown", "task_unlisted", 'A task directory has not been fully listed', source=directory["path"])
            continue
        if not child.complete:
            collector.add("10.2", "unknown", "task_partial", 'A task directory listing is incomplete', source=directory["path"], evidence=child.reason)
        if not child.entries and child.complete:
            collector.add("10.2", "fail", "empty_task", 'A task directory is empty', source=directory["path"])
        nested = [e for e in child.entries if e["kind"] == "directory"]
        if nested:
            collector.add("10.2", "fail", "nested_task", 'A task directory contains nested directories', source=directory["path"], evidence=nested[:5])
        levels.append((directory["path"], [e for e in child.entries if e["kind"] == "file"]))
    for parent, files_in_parent in levels:
        for item in files_in_parent:
            kind = item.get("archive_kind", archive_kind(item["path"]))
            if kind in ("plain", "pointer"):
                collector.add("10.2", "fail", "non_archive" if kind == "plain" else "lfs_pointer", "The directory contains a non-archive file or an LFS pointer", source=item["path"])
            elif kind == "unknown":
                collector.add("10.2", "unknown", "archive_unrecognized", 'Archive format could not be identified; formats are unrestricted', source=item["path"])
            if item.get("size") == 0:
                collector.add("10.2", "fail", "empty_archive", 'A data file has zero size', source=item["path"])
            elif item.get("size") is None:
                collector.add("10.2", "unknown", "size_unknown", 'File size is unavailable', source=item["path"])
    return "B" if directories else "A"


def probe_video(path: Path) -> dict:
    if not path.is_file():
        return {"error": "File is missing"}
    executable = shutil.which("ffprobe")
    if not executable:
        return {"unknown": "ffprobe is unavailable"}
    try:
        process = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=width,height,codec_name:format=duration", "-of", "json", str(path)],
                                 capture_output=True, text=True, timeout=20)
        if process.returncode:
            return {"error": process.stderr.strip()[:250]}
        obj = json.loads(process.stdout)
        stream = obj.get("streams", [{}])[0]
        if not stream.get("width") or not stream.get("height"):
            return {"error": "No valid video stream"}
        return {"width": stream["width"], "height": stream["height"], "codec": stream.get("codec_name"),
                "duration": obj.get("format", {}).get("duration")}
    except Exception as error:
        return {"unknown": f"Video probing failed: {type(error).__name__}"}




def validate_metadata(folder: Path, meta: dict, c: Collector):
    c.applicable.add("10")
    if missing := [key for key in FIELDS if key not in meta]:
        c.add("10", "fail", "missing_fields", 'Required metadata fields are missing', source="metadata.json", evidence=missing)
    for key in ("name", "version", "description"):
        if not isinstance(meta.get(key), str) or not meta[key].strip():
            c.add("10", "fail", "unfilled", 'Metadata fields are empty or are not strings', key=key, source="metadata.json")
    if "code_trigger" in meta and meta["code_trigger"] != "":
        c.add("10", "fail", "code_trigger_not_empty", "code_trigger must be an empty string", key="code_trigger", source="metadata.json", evidence=meta["code_trigger"])
    for key in ("url", "sensor_image", "sensor_array", "sensor_state", "label", "contact"):
        value = meta.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            c.add("10", "fail", "field_type", 'Metadata field types are invalid', key=key, source="metadata.json", evidence=type(value).__name__)
    if not meta.get("url"):
        c.add("10", "fail", "url_empty", 'A download URL is missing', source="metadata.json")
    elif isinstance(meta["url"], list) and len(meta["url"]) != 1:
        c.add("10", "fail", "url_count", "Exactly one download URL is required", key="url", source="metadata.json", evidence={"count": len(meta["url"]), "urls": meta["url"]})
    if meta.get("type") not in ("gripper", "dexhand", "human"):
        c.add("10", "fail", "embodiment_type", 'Metadata.type must be gripper/dexhand/human', source="metadata.json", evidence=meta.get("type"))
    for key in ("task", "trajectory", "frame"):
        value = meta.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            c.add("10", "fail", "count_type", 'Metadata counts must be positive integers', key=key, source="metadata.json", evidence=value)
    labels = set(strings(meta.get("label")))
    if labels - LABELS:
        c.add("10", "fail", "labels", 'Labels other than ego/umi/with-supp are present', source="metadata.json", evidence=sorted(labels - LABELS))
    for path, value in walk_strings(meta):
        if CJK_RE.search(value) or CJK_RE.search(path):
            c.add("10", "fail", "metadata_chinese", 'Metadata contains Chinese text', key=path, source="metadata.json", evidence=value[:200])
    contacts = strings(meta.get("contact"))
    if not contacts or any(not EMAIL_RE.fullmatch(v) for v in contacts):
        c.add("10", "fail", "contact_email", 'Contact must contain email addresses only', source="metadata.json", evidence=contacts)
    if not isinstance(meta.get("reference"), dict):
        c.add("10", "fail", "reference_type", 'Reference must be an object', source="metadata.json")


def check_assets(folder: Path, meta: dict, labels: set[str], sensors: set[str], c: Collector) -> dict:
    c.applicable.update(("06", "08"))
    try:
        from PIL import Image
    except ImportError:
        Image = None
        c.add("06", "unknown", "pillow_missing", 'Pillow is unavailable; image decoding could not be checked')
    images = {p.name: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg")}
    for name in ("hand_joints_idx.png", "tactile_func_area_idx.png"):
        if name not in images:
            alternatives = [n for n in images if name.rsplit(".", 1)[0] in n]
            c.add("06", "fail", "annotation_missing", 'A required annotation image is missing', key=name, evidence=alternatives)
    if not any(n.startswith("institution_") and n.endswith(".png") for n in images):
        c.add("06", "fail", "institution_missing", 'An institution image is missing')
    if not any(n.startswith("sensor_") and n.endswith(".png") for n in images):
        c.add("06", "fail", "sensor_missing", 'A sensor image is missing')
    for name, image in images.items():
        if " " in name and name.startswith(("institution_", "sensor_")):
            c.add("06", "fail", "image_filename", 'An image filename contains spaces', source=name)
        if Image and image.suffix.lower() != ".svg":
            try:
                with Image.open(image) as opened:
                    opened.verify()
            except Exception as error:
                c.add("06", "fail", "image_invalid", 'An image cannot be decoded', source=name, evidence=type(error).__name__)
    declared = set(v for field in ("sensor_image", "sensor_array", "sensor_state") for v in strings(meta.get(field)))
    for sensor in sorted(declared | sensors):
        if "sensor_" + sensor + ".png" not in images:
            matches = [n for n in images if n.casefold() == ("sensor_" + sensor + ".png").casefold()]
            c.add("12", "warning", "sensor_photo_name", 'A sensor has no PNG with the exact matching name', key=sensor, evidence=matches)
    example_video = probe_video(folder / "video.mp4")
    if "error" in example_video:
        c.add("08", "fail", "video_missing_or_invalid", 'A valid video.mp4 is missing', source="video.mp4", evidence=example_video)
    elif "unknown" in example_video:
        c.add("08", "unknown", "video_unverified", 'Video format could not be verified', source="video.mp4", evidence=example_video)
    elif example_video["width"] != example_video["height"]:
        c.add("08", "fail", "video_square", 'Video.mp4 is not square', source="video.mp4", evidence=example_video)
    if labels & {"ego", "umi"}:
        c.applicable.add("04")
        viser = probe_video(folder / "4d_viser_check.mp4")
        if "error" in viser:
            c.add("04", "fail", "viser_missing_or_invalid", 'A valid Viser 4D recording is missing', source="4d_viser_check.mp4", evidence=viser)
        elif "unknown" in viser:
            c.add("04", "unknown", "viser_unverified", 'Viser recording format could not be verified', evidence=viser)
    return {"example_video": example_video, "images": sorted(images)}


def check_arrays(folder: Path, meta: dict, dict_obj: Any, c: Collector) -> dict:
    metrics = {"tables": 0, "named_tables": 0, "flat_schema": False, "frames": 0, "episodes": 0,
               "missing_episode_ends": 0, "sensors": set(), "sensor_types": defaultdict(set),
               "supplementary_shapes": defaultdict(set), "compression": Counter(), "no_actions": 0,
               "ego_rgb_missing": 0, "ego_pose_missing": 0}
    embodiment_type = meta.get("type")
    for table_path, table in iter_tables(dict_obj):
        metrics["tables"] += 1
        metrics["named_tables"] += bool(table_path)
        metrics["flat_schema"] |= not bool(table_path)
        example = "/".join(table_path) or "<flat schema>"
        valid = {}
        for key, info in table.items():
            shape = info["shape"]
            if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape):
                c.add("02", "fail", "invalid_shape", 'Shape contains invalid dimensions', key=key, example=example, evidence=shape)
                continue
            valid[key] = info
        table = valid
        T = table.get("timestamps", {}).get("shape", [None])[0] if table.get("timestamps", {}).get("shape") else None
        c.applicable.update(("03", "09", "12"))
        if T is None:
            c.add("03", "fail", "timestamps_missing", 'Timestamps are missing', example=example)
        else:
            metrics["frames"] += T
        ep = table.get("meta/episode_ends") or table.get("episode_ends")
        if ep and len(ep["shape"]) == 1:
            metrics["episodes"] += ep["shape"][0]
            values = flatten(ep.get("constant_value"))
            if values and (any(not isinstance(v, int) or v <= 0 for v in values) or any(b <= a for a, b in zip(values, values[1:])) or values[-1] != T):
                c.add("03", "fail", "episode_ends_values", 'Episode ends are not increasing or the final boundary is not T', example=example, evidence=values)
        else:
            metrics["missing_episode_ends"] += 1
            c.add("03", "unknown", "episode_ends_missing", 'Usable episode_ends are unavailable', example=example)
        acts = action_keys(table)
        if not acts:
            metrics["no_actions"] += 1
            c.add("03", "fail", "no_actions", 'Joint/pose/hand action information is missing', example=example)
        if "sub_task_instruction" not in table:
            c.add("09", "fail", "instruction_missing", 'Sub_task_instruction is missing', example=example)
        else:
            info = table["sub_task_instruction"]
            if info["shape"] != [T] or not any(s in str(info.get("dtype", "")) for s in ("U", "S", "str", "object")):
                c.add("09", "fail", "instruction_schema", 'Sub_task_instruction must be a (T,) string array', example=example, evidence=info)
            if info.get("constant_value") is not None and not str(info["constant_value"]).strip():
                c.add("09", "fail", "instruction_empty", 'Sub_task_instruction is empty', example=example)
        if not any(key.endswith("_rgb") for key in table):
            c.add("01", "fail", "rgb_missing", 'RGB data is missing', example=example)
        metrics["ego_rgb_missing"] += "camera_ego_rgb" not in table
        metrics["ego_pose_missing"] += "camera_ego_pose" not in table
        groups = defaultdict(dict)
        for key, info in table.items():
            shape = info["shape"]
            ev = {"shape": shape, "dtype": info.get("dtype")}
            non_time = shape[1:]
            data_key = not key.startswith("meta/") and key != "episode_ends"
            if not key.startswith("meta/") and key != "episode_ends" and shape and T is not None and shape[0] != T:
                c.add("03", "fail", "frame_mismatch", 'Data frame count does not match timestamps', key=key, example=example, evidence={**ev, "T": T}, signature=non_time)
            if data_key and key.endswith("_rgb"):
                c.applicable.add("01")
                if len(shape) != 4 or shape[-1] not in (3, 4) or str(info.get("dtype")) not in UINT8:
                    c.add("01", "fail", "rgb_schema", 'RGB must have shape (T,H,W,3) and dtype uint8', key=key, example=example, evidence=ev, signature=non_time)
            if data_key and POSE_RE.match(key):
                c.applicable.add("05")
                if non_time != [6]:
                    c.add("05", "fail", "standard_pose_shape", 'Standard pose must have six dimensions', key=key, example=example, evidence=ev, signature=non_time)
            if data_key and key.endswith("_hand_joints"):
                idx = table.get(key + "_idx")
                static = table.get("meta/action_slot_idx__" + key)
                if idx is None and static is None:
                    c.add("03", "fail", "joint_idx_missing", 'Hand-joint indices are missing', key=key, example=example)
                elif idx is not None and idx["shape"] != shape:
                    c.add("03", "fail", "joint_idx_shape", 'Hand-joint data and index shapes do not match', key=key, example=example, evidence={**ev, "idx_shape": idx["shape"]}, signature=non_time)
                elif static is not None and (len(shape) != 2 or static["shape"] != [shape[1]]):
                    c.add("03", "fail", "static_idx_shape", 'Static action index count does not match the channel count', key=key, example=example, evidence=static)
                if idx is not None and static is not None:
                    local = flatten(idx.get("constant_value"))
                    global_values = flatten(static.get("constant_value"))
                    offset = 16 if key.startswith("right_") else 64 if key.startswith("left_") else None
                    if local and global_values and offset is not None and all(isinstance(v, int) for v in local) and [v + offset for v in local] != global_values:
                        c.add("03", "fail", "legacy_static_idx_conflict", 'Legacy and static hand-joint indices disagree', key=key, evidence={"local": local, "global": global_values})
            local_hand = key.endswith("_hand_joints_idx")
            local_area = "_tactile_area_" in key and not key.startswith("supplementary")
            global_action = key.startswith("meta/action_slot_idx__")
            global_area = key.startswith("meta/tactile_slot_idx__")
            if local_hand or local_area or global_action or global_area:
                values = flatten(info.get("constant_value"))
                limit = 119 if global_action else 47 if global_area else 31 if local_hand else 23
                if not str(info.get("dtype", "")).startswith(("int", "uint")):
                    c.add("03", "fail", "idx_dtype", 'Indices must have an integer dtype', key=key, evidence=ev)
                if values and any(not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= limit for v in values):
                    c.add("03", "fail", "idx_range", 'Indices are outside the permitted range', key=key, example=example, evidence=values[:40], signature=values[:40])
                if values and len(values) != len(set(map(str, values))):
                    c.add("03", "fail", "idx_duplicate", 'Indices contain duplicates', key=key, example=example, evidence=values[:40], signature=values[:40])
            if info.get("is_constant_over_time") is False:
                c.add("03", "fail", "constant_changed", 'Fixed indices, names, or types change over time', key=key, example=example)
            match = TACTILE_RE.match(key) if data_key else None
            if match:
                groups[(match[1], match[3])][match[2]] = (key, info)
            if data_key and "tactile_sensor_" in key:
                values = strings(info.get("constant_value"))
                metrics["sensors"].update(values)
                tk = key.replace("tactile_sensor_", "tactile_type_", 1)
                for sensor in values:
                    metrics["sensor_types"][sensor].update(strings(table.get(tk, {}).get("constant_value")))
            if key.startswith("supplementary"):
                metrics["supplementary_shapes"][key].add(tuple(non_time))
            if data_key and (key.endswith(("_rgb", "_depth", "_image")) or len(shape) >= 4):
                compression = "unrecorded" if "compression" not in info else "uncompressed" if info["compression"] in (None, []) else "compressed"
                metrics["compression"][compression] += 1
                if compression == "uncompressed":
                    c.add("11", "warning", "image_uncompressed", 'Exported image compression is disabled', key=key, example=example, signature=non_time)
        pad_areas = defaultdict(set)
        pad_counts = defaultdict(int)
        pad_indices = {}
        if embodiment_type == "gripper":
            for (prefix, detail), group in groups.items():
                data = group.get("data", (None, {}))[1]
                shape = data.get("shape", [])
                if prefix == "supplementary" or len(shape) < 2 or shape[1] not in (1, 2):
                    continue
                data_key = prefix + "_tactile_data_" + detail
                index = group.get("area")
                offset = 0
                if index is None:
                    static_key = "meta/tactile_slot_idx__" + data_key
                    if static_key not in table:
                        continue
                    index = (static_key, table[static_key])
                    if not prefix.startswith(("left", "right")):
                        c.add("03", "unknown", "gripper_area_unrecorded", "Gripper local area could not be resolved from the static indices", key=static_key, example=example)
                        continue
                    offset = 24 if prefix.startswith("left") else 0
                values = flatten(index[1].get("constant_value"))
                if not values:
                    c.add("03", "unknown", "gripper_area_unrecorded", "Gripper area constants are unavailable", key=index[0], example=example)
                    continue
                if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                    continue
                local = [v - offset for v in values]
                pad_areas[prefix].update(local)
                pad_counts[prefix] = max(pad_counts[prefix], shape[1])
                pad_indices[data_key] = (index[0], values, local, offset)
        for (prefix, detail), members in groups.items():
            c.applicable.add("02")
            required = {"data", "sensor", "type"} if prefix == "supplementary" else {"data", "area", "sensor", "type"}
            dk = prefix + "_tactile_data_" + detail
            static_area = table.get("meta/tactile_slot_idx__" + dk)
            if static_area:
                required.discard("area")
            if missing := required - set(members):
                c.add("02", "fail", "tactile_members", 'Required tactile group members are missing', key=dk, example=example, evidence=sorted(missing))
            if "data" not in members:
                continue
            key, info = members["data"]
            shape = info["shape"]
            ev = {"shape": shape, "dtype": info.get("dtype")}
            if static_area is not None and (len(shape) < 2 or static_area["shape"] != [shape[1]]):
                c.add("02", "fail", "static_area_shape", 'Static tactile index count does not match data dimension N', key=key, example=example, evidence=static_area)
            type_info = members.get("type", (None, {}))[1]
            types = set(strings(type_info.get("constant_value")))
            ev["types"] = sorted(types)
            if not types:
                c.add("02", "unknown", "type_unrecorded", 'The tactile type has no readable constant value', key=key, example=example)
            elif types - TYPES:
                c.add("02", "fail", "invalid_tactile_type", 'Tactile type is invalid', key=key, example=example, evidence=sorted(types - TYPES))
            if "image" in types:
                c.applicable.add("01")
                if str(info.get("dtype")) not in UINT8:
                    c.add("01", "fail", "image_not_uint8", IMAGE_DTYPE_GUIDANCE, key=key, example=example, evidence=ev, signature=shape[1:])
            if "state" in types and len(shape) != 3:
                c.add("02", "fail", "state_shape", "State tactile data must be three-dimensional; use matrix for spatial grids", key=key, example=example, evidence=ev, signature=shape[1:])
            dtype = str(info.get("dtype", "")).lower()
            floating = bool(re.fullmatch(r"(?:float(?:16|32|64|128)?|bfloat16|[<>=|]?f(?:2|4|8|16))", dtype))
            if floating and len(shape) in (4, 5) and types and "matrix" not in types and "state" not in types:
                c.add("02", "fail", "floating_grid_type", "Floating-point spatial tactile data must use type=matrix", key=key, example=example, evidence=ev, signature=shape[1:])
            if "area" in members:
                area = members["area"][1]
                if len(shape) < 2 or area["shape"] != shape[:2]:
                    c.add("02", "fail", "tactile_area_shape", 'Tactile data and area have different T/N dimensions', key=key, example=example, evidence={**ev, "area_shape": area["shape"]}, signature=shape[1:])
                area_values = flatten(area.get("constant_value"))
                if static_area is not None:
                    global_values = flatten(static_area.get("constant_value"))
                    offset = 24 if prefix == "left" else 0 if prefix == "right" else None
                    if area_values and global_values and offset is not None and all(isinstance(v, int) for v in area_values) and [v + offset for v in area_values] != global_values:
                        c.add("03", "fail", "legacy_static_area_conflict", 'Legacy and static tactile indices disagree', key=key, evidence={"local": area_values, "global": global_values})
                if len(area_values) > 1 and len(set(map(str, area_values))) == 1:
                    c.add("02", "fail", "functional_area_not_consolidated", 'One functional area occupies duplicate channels', key=key,
                          example=example, evidence={**ev, "area": area_values[:30], "target": "area=(T,1), data=(T,1,D)"}, signature=shape[1:])
            if key in pad_indices:
                # Separate N=1 streams can still represent the two pads of one gripper.
                index_key, actual, local, offset = pad_indices[key]
                count = min(2, max(pad_counts[prefix], len(pad_areas[prefix])))
                target = list(range(count))
                if target is not None and not set(local).issubset(target):
                    c.add("03", "fail", "gripper_area", "Gripper local area must be 0 or [0,1]", key=index_key, example=example,
                          evidence={"actual": actual, "expected": [v + offset for v in target], "local_actual": local, "local_expected": target}, signature=actual)
            for member in ("sensor", "type"):
                if member in members and members[member][1]["shape"] != [T]:
                    c.add("02", "fail", "tactile_string_shape", 'Sensor/type must be a (T,) string array', key=members[member][0], example=example, evidence=members[member][1]["shape"])
    if not metrics["tables"]:
        for rule in ("01", "02", "03", "05", "09", "12", "13", "14"):
            c.add(rule, "unknown", "no_tables", 'No data tables could be parsed from dict')
    if metrics["tables"] and not metrics["flat_schema"]:
        if metrics["frames"] != meta.get("frame"):
            c.add("10", "fail", "frame_count", 'Metadata.frame does not match the exported frame count', evidence={"metadata": meta.get("frame"), "dict": metrics["frames"]})
        if not metrics["missing_episode_ends"] and metrics["episodes"] != meta.get("trajectory"):
            c.add("10", "fail", "trajectory_count", 'Metadata.trajectory does not match the episode count', evidence={"metadata": meta.get("trajectory"), "dict": metrics["episodes"]})
    else:
        c.add("10", "unknown", "statistics_scope", 'A flat or missing schema cannot establish complete episode statistics')
    if metrics["compression"]["unrecorded"]:
        c.add("11", "unknown", "compression_unrecorded", 'Compression metadata is missing for some arrays', evidence=dict(metrics["compression"]))
    return metrics


def check_supplementary(folder: Path, metrics: dict, labels: set[str], c: Collector):
    c.applicable.add("07")
    shapes = metrics.get("supplementary_shapes", {})
    data_keys = [k for k in shapes if k == "supplementary_joints" or k == "supplementary_tactile" or k.startswith("supplementary_tactile_data_") or k == "supplementary_action_commands"]
    notes = sorted(folder.glob("supplementary*_note*.json"))
    any_supp_files = [p.name for p in folder.iterdir() if p.is_file() and p.name.startswith("supplementary_")]
    if bool(any_supp_files) != ("with-supp" in labels):
        c.add("07", "warning", "supp_file_label", 'Supplementary files and the with-supp label disagree', evidence={"files": any_supp_files, "label": sorted(labels)})
    if data_keys and "with-supp" not in labels:
        c.add("07", "fail", "supp_data_without_label", 'Supplementary data exists without the with-supp label', evidence=data_keys)
    elif not data_keys and "with-supp" in labels:
        status = "warning" if metrics.get("tables") else "unknown"
        c.add("07", status, "supp_label_without_data", 'The with-supp label exists but no supplementary arrays were exported', evidence={"notes": [p.name for p in notes]})
    if data_keys or notes:
        c.applicable.add("13")
    for key, expected_note in (("supplementary_joints", "supplementary_joints_note.json"), ("supplementary_tactile", "supplementary_tactile_note.json")):
        present = key in shapes or (key == "supplementary_tactile" and any(k.startswith("supplementary_tactile_data_") for k in shapes))
        if present and not (folder / expected_note).is_file():
            c.add("13", "fail", "required_note_missing", 'A required supplementary JSON note is missing', key=key, source=expected_note)
    joint_dimensions = {s[0] for s in shapes.get("supplementary_joints", []) if len(s) == 1}
    described_dimensions = set()
    for note in notes:
        try:
            obj = read_json(note)
        except Exception as error:
            c.add("13", "fail", "note_json", 'A supplementary note is unreadable', source=note.name, evidence=type(error).__name__)
            continue
        if not isinstance(obj, dict):
            c.add("13", "fail", "note_object", 'A supplementary note must be a JSON object', source=note.name)
            continue
        for key, entry in obj.items():
            if not isinstance(entry, dict):
                continue
            dim = entry.get("dimension")
            if dim is not None:
                if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                    c.add("13", "fail", "note_dimension", 'Supplementary dimension must be a positive integer', key=key, source=note.name)
                else:
                    described_dimensions.add(dim)
                    desc = entry.get("dim_description", {})
                    if not isinstance(desc, dict) or any(not isinstance(desc.get(f"dim_{i}"), str) or not desc[f"dim_{i}"].strip() for i in range(1, dim + 1)):
                        c.add("13", "fail", "note_dimensions_incomplete", 'Supplementary dimension descriptions are incomplete', key=key, source=note.name)
            n, shape, dims = entry.get("sensor_num"), entry.get("shape"), entry.get("sensor_dimension")
            if isinstance(n, int) and isinstance(shape, list) and isinstance(dims, (int, list)):
                per_sensor = dims if isinstance(dims, list) else [dims]
                if shape != [n] + per_sensor and not (n == 1 and shape == per_sensor):
                    c.add("13", "fail", "note_shape_inconsistent", 'Supplementary shape disagrees with sensor count and dimensions', key=key, source=note.name,
                          evidence={"shape": shape, "expected": [n] + per_sensor})
            descriptions = str(entry.get("sensor_description", ""))
            if re.search(r"please use english|TODO|填入|待填写", descriptions, re.I):
                c.add("13", "fail", "note_placeholder", 'A supplementary note still contains template placeholders', key=key, source=note.name)
    if joint_dimensions - described_dimensions:
        c.add("13", "fail", "unexplained_joint_dimensions", 'Some supplementary joint dimensions have no description', evidence=sorted(joint_dimensions - described_dimensions))


def check_existing_qc(folder: Path, c: Collector) -> dict:
    c.applicable.add("11")
    qdir = folder / "oxt_check_output"
    result = {"files": [], "precheck_warnings": {}, "vlm_counts": {}}
    if not qdir.is_dir():
        c.add("11", "fail", "qc_dir_missing", 'The oxt_check_output directory is missing')
    pre_path, vlm_path = qdir / "precheck_report_result.json", qdir / "vlm_review_result.json"
    for path in (pre_path, vlm_path):
        if not path.is_file():
            c.add("11", "fail", "qc_result_missing", 'Required QC result files are missing', source=str(path.relative_to(folder)))
        else:
            result["files"].append(str(path.relative_to(folder)))
    if pre_path.is_file():
        try:
            pre = read_json(pre_path)
            if not isinstance(pre, dict):
                raise ValueError("The precheck root is not an object")
            counts = Counter()
            for dataset in pre.get("datasets", []):
                if not isinstance(dataset, dict):
                    continue
                for check in dataset.get("failed_checks", []):
                    if not isinstance(check, dict):
                        continue
                    if "supplementary/" in str(check.get("name")) and "missing: area" in str(check.get("message")):
                        c.add("11", "warning", "supp_area_tool_conflict", 'Legacy QC requires supplementary area although the guide permits three members', source=str(pre_path.relative_to(folder)))
                    else:
                        c.add("11", "warning", "precheck_reported_fail", 'An existing precheck result records a fail', key=str(check.get("name", "")),
                              source=str(pre_path.relative_to(folder)), example=str(dataset.get("dataset", "")), evidence=check)
                for check in dataset.get("warning_checks", []):
                    if isinstance(check, dict):
                        counts[str(check.get("name", "unknown"))] += 1
                        c.add("11", "warning", "precheck_warning", 'An existing precheck result records a warning', key=str(check.get("name", "")),
                              source=str(pre_path.relative_to(folder)), example=str(dataset.get("dataset", "")), evidence=str(check.get("message", ""))[:500])
            result["precheck_warnings"] = dict(counts)
        except Exception as error:
            c.add("11", "fail", "precheck_unreadable", 'The precheck result is unreadable', source=str(pre_path.relative_to(folder)), evidence=type(error).__name__)
    if vlm_path.is_file():
        try:
            vlm = read_json(vlm_path)
            if not isinstance(vlm, dict):
                raise ValueError("The VLM root is not an object")
            reports = vlm.get("reports", [])
            if not isinstance(reports, list):
                raise ValueError("Reports must be a list")
            page_counts = Counter()
            error_reasons = []
            for report in reports:
                if not isinstance(report, dict):
                    continue
                for aspect in ASPECTS:
                    block = report.get(aspect)
                    if not isinstance(block, dict):
                        continue
                    numeric = []
                    for status in ("pass", "warning", "fail", "error"):
                        pages = block.get(status + "_pages", [])
                        if not isinstance(pages, list):
                            continue
                        page_counts[status] += len(pages)
                        for page in pages:
                            if not isinstance(page, dict):
                                continue
                            score = page.get("score")
                            if status != "error" and (not isinstance(score, (int, float)) or isinstance(score, bool)):
                                c.add("11", "unknown", "vlm_score_missing", 'A non-error VLM page has no valid numeric score', key=aspect)
                            if isinstance(score, (int, float)) and not isinstance(score, bool):
                                if not math.isfinite(score) or not 0 <= score <= 1:
                                    c.add("11", "fail", "vlm_score_invalid", 'VLM scores must be within [0,1]', key=aspect, evidence=str(score))
                                    continue
                                numeric.append(score)
                                expected = "pass" if score >= .8 else "warning" if score >= .5 else "fail"
                                if status != "error" and status != expected:
                                    c.add("11", "fail", "vlm_bucket", 'VLM scores do not match their pass/warning/fail buckets', key=aspect, evidence={"score": score, "bucket": status})
                            if status == "error":
                                error_reasons.append(str(page.get("reason", "")))
                            if status != "pass":
                                c.add("11", "warning", "vlm_reported_" + status, 'An existing VLM result records a finding', key=aspect,
                                      source=str(vlm_path.relative_to(folder)), example=str(report.get("dataset", "")),
                                      evidence={"page": page.get("page_index"), "score": score, "reason": strings(page.get("reason"))[:3]})
                    average = block.get("average_score")
                    if numeric and isinstance(average, (int, float)) and abs(sum(numeric) / len(numeric) - average) > 1e-7:
                        c.add("11", "fail", "vlm_average", 'The VLM average score is inconsistent', key=aspect,
                              evidence={"stored": average, "computed": sum(numeric) / len(numeric)})
            result["vlm_counts"] = dict(page_counts)
            not_run = vlm.get("status") == "not_run" or (error_reasons and not sum(page_counts[k] for k in ("pass", "warning", "fail")) and all(re.search(r"dry_run_no_api_call|missing_api_key|no_api", s, re.I) for s in error_reasons))
            if not_run:
                c.add("11", "fail", "vlm_not_run", 'VLM was not actually run', source=str(vlm_path.relative_to(folder)))
            elif not reports:
                c.add("11", "unknown", "vlm_no_reports", 'The VLM result has no parseable review records', source=str(vlm_path.relative_to(folder)))
            if reports and not page_counts:
                c.add("11", "unknown", "vlm_aspect_schema", 'The VLM result has no parseable four-aspect page structure')
        except Exception as error:
            c.add("11", "fail", "vlm_unreadable", 'The VLM result is unreadable', source=str(vlm_path.relative_to(folder)), evidence=type(error).__name__)
    return result


def check_remote(meta: dict, options: Options, client: RemoteClient, c: Collector) -> list[dict]:
    parsed = []
    for url in strings(meta.get("url")):
        try:
            parsed.append((url, parse_source(url)))
        except Exception as error:
            c.add("10.1", "fail", "dataset_url", 'The download URL is not a valid HF/MS dataset URL', source=url, evidence=str(error))
    sources = {}
    for url, source in parsed:
        sources.setdefault((source.provider, source.repo, source.revision), (url, source))
        c.applicable.add("10.1")
        name = source.repo.rsplit("/", 1)[-1]
        if not name.endswith("_OXT"):
            c.add("10.1", "fail", "repo_suffix", "Repository name must end with _OXT", source=url, evidence={"expected_suffix": "_OXT", "repo": source.repo})
        if source.filename or (source.subdir and source.subdir != name):
            c.add("10.1", "fail", "partial_entry", 'The download URL points to a subdirectory or a single file', source=url, evidence=asdict(source))
    if not parsed:
        c.add("10.2", "unknown", "no_remote_entry", 'No readable remote dataset URL is available')
    result = []
    for url, source in sources.values():
        name = source.repo.rsplit("/", 1)[-1]
        if options.offline:
            c.add("10.2", "unknown", "offline", 'Remote checks were skipped in offline mode', source=url)
            if source.provider == "hf":
                c.add("10.3", "unknown", "offline", 'Remote checks were skipped in offline mode', source=url)
            continue
        remote = dict(client.root(source))
        remote["url"] = url
        result.append(remote)
        if "error" in remote:
            c.add("10.2", "unknown", "remote_unreachable", 'The remote directory could not be read', source=url, evidence=remote["attempts"])
            if source.provider == "hf":
                c.add("10.3", "unknown", "hf_and_mirror_unreachable", 'Neither HF nor the mirror completed the directory read', source=url, evidence=remote["attempts"])
            continue
        if source.provider == "hf":
            c.applicable.add("10.3")
            if remote["endpoint"] == options.mirror_endpoint:
                c.add("10.3", "pass", "mirror_used", 'The same repository and revision were checked through hf-mirror', source=url, evidence={"endpoint": remote["endpoint"], "revision": remote["revision"]})
                if not remote.get("official_sha") and not re.fullmatch(r"[0-9a-fA-F]{40}", source.revision or ""):
                    remote["official_sync_unverified"] = True
                    c.add("10.3", "unknown", "mirror_sync_unverified", 'The mirror was read, but its synchronization with the official branch is unverified', source=url, evidence=remote["revision"])
        root = remote["root"]
        if not root["complete"]:
            c.add("10.2", "unknown", "root_partial", 'The repository root listing is incomplete', source=url, evidence=root["reason"])
        readme = next((e for e in root["entries"] if PurePosixPath(e["path"]).name.casefold() == "readme.md" and e["kind"] == "file"), None)
        deadline = time.monotonic() + options.repo_budget

        def file_read(path: str, limit: int = 65536):
            try:
                return client.file(remote["endpoint"], source, remote["revision"], path, deadline, limit=limit)
            except RemoteError:
                if source.provider != "hf" or remote["endpoint"] == options.mirror_endpoint:
                    raise
                data = client.file(options.mirror_endpoint, source, remote["revision"], path, deadline, limit=limit)
                remote.setdefault("fallback_paths", []).append(path)
                c.add("10.3", "pass", "mirror_file_used", 'HF file access failed; the mirror served the same revision and path', source=url, evidence=path)
                return data

        if readme:
            try:
                data = file_read(readme["path"])
                if b"open-x-tactile" not in data.lower():
                    c.add("10.1", "warning", "readme_oxt", 'No Open-X-Tactile reference was found in the bounded README text', source=url)
            except RemoteError as error:
                c.add("10.1", "unknown", "readme_unverified", 'README content could not be read', source=url, evidence=str(error))
        elif root["complete"]:
            c.add("10.1", "fail", "readme_missing", 'README.md is missing from the repository root', source=url)
        target = next((e for e in root["entries"] if e["path"] == name and e["kind"] == "directory"), None)
        if not target:
            c.add("10.2", "fail" if root["complete"] and not remote.get("official_sync_unverified") else "unknown", "dataset_dir_missing", 'The repository lacks the exact DatasetName directory', source=url,
                  evidence={"expected": name, "root_preview": [e["path"] for e in root["entries"][:15]], "complete": root["complete"]})
            continue
        def list_tree(parent: str):
            if source.provider == "ms":
                return client.list_ms(source, parent, deadline)
            try:
                return client.list_hf(remote["endpoint"], source, remote["revision"], parent, deadline)
            except RemoteError:
                if remote["endpoint"] == options.mirror_endpoint:
                    raise
                tree = client.list_hf(options.mirror_endpoint, source, remote["revision"], parent, deadline)
                remote.setdefault("fallback_paths", []).append(parent)
                c.add("10.3", "pass", "mirror_subtree_used", 'HF subtree access failed; the mirror served the same revision and path', source=url, evidence=parent)
                return tree
        try:
            tree = list_tree(name)
            children = {}
            directories = [e for e in tree.entries if e["kind"] == "directory"]
            if len(directories) > options.max_tasks:
                c.add("10.2", "unknown", "task_limit", 'The number of task directories exceeds the check limit', source=url)
            for entry in directories[:options.max_tasks]:
                try:
                    children[entry["path"]] = list_tree(entry["path"])
                except RemoteError as error:
                    children[entry["path"]] = Tree(complete=False, reason=str(error))
            # Identify unknown formats and sample actual archive/pointer headers.
            files = [e for e in tree.entries if e["kind"] == "file"] + [e for child in children.values() for e in child.entries if e["kind"] == "file"]
            sampled = 0
            for entry in files:
                kind = archive_kind(entry["path"])
                if kind == "unknown" or (kind == "archive" and sampled < options.archive_samples and not re.search(r"\.\d{2,}$", entry["path"])):
                    try:
                        prefix = file_read(entry["path"], limit=512)
                        entry["archive_kind"] = archive_kind(entry["path"], prefix)
                        sampled += 1
                    except RemoteError as error:
                        entry["archive_kind"] = "unknown"
                        c.add("10.2", "unknown", "archive_sample_unavailable", 'The archive header sample could not be read', source=entry["path"], evidence=str(error))
            remote["layout"] = check_layout(tree.entries, name, children, c, complete=tree.complete)
            remote["dataset_tree"] = {"complete": tree.complete, "pages": tree.pages, "entries": tree.entries[:options.max_entries]}
            remote["tasks"] = {p: asdict(t) for p, t in children.items()}
            remote["archive_header_samples"] = sampled
            if directories and tree.complete and len(directories) != meta.get("task"):
                c.add("10.2", "warning", "task_folder_count", 'The task directory count differs from metadata.task', source=url,
                      evidence={"folders": len(directories), "metadata_task": meta.get("task")})
        except RemoteError as error:
            c.add("10.2", "unknown", "dataset_tree_unavailable", 'The dataset directory was not fully read', source=url, evidence=str(error))
    return result




def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def review_dataset(folder: Path, options: Options, client: RemoteClient) -> dict:
    folder = folder.resolve()
    c = Collector()
    initial_stats = {}
    sources = {}
    for name in ("metadata.json", "oxt_check_output/dict.json", "oxt_check_output/precheck_report_result.json", "oxt_check_output/vlm_review_result.json"):
        path = folder / name
        if path.is_file():
            stat = path.stat()
            initial_stats[name] = (stat.st_size, stat.st_mtime_ns)
            sources[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    meta = {}
    metadata_loaded = False
    try:
        meta = read_json(folder / "metadata.json")
        if not isinstance(meta, dict):
            raise ValueError("The metadata root is not an object")
        metadata_loaded = True
        sources["metadata.json"]["sha256"] = hash_file(folder / "metadata.json")
    except Exception as error:
        c.add("10", "fail", "metadata_unreadable", 'Metadata.json is missing or unreadable', evidence=f"{type(error).__name__}: {error}")
        for rule in RULES:
            if rule != "10":
                c.add(rule, "unknown", "metadata_unavailable", 'Metadata is unavailable; related checks are incomplete')
        meta = {}
    metrics: dict = {"tables": 0, "sensors": set(), "supplementary_shapes": {}, "compression": {}}
    assets = {}
    qc = {}
    remote = []
    if metadata_loaded:
        validate_metadata(folder, meta, c)
        labels = set(strings(meta.get("label")))
        dict_path = folder / "oxt_check_output/dict.json"
        try:
            dict_obj = read_json(dict_path)
            metrics = check_arrays(folder, meta, dict_obj, c)
            sources["oxt_check_output/dict.json"]["sha256"] = hash_file(dict_path)
        except Exception as error:
            for rule in ("01", "02", "03", "05", "09", "12", "13", "14"):
                c.add(rule, "unknown", "dict_unavailable", 'Dict.json is unavailable or cannot be parsed', source="oxt_check_output/dict.json", evidence=f"{type(error).__name__}: {error}")
            c.add("11", "fail", "dict_missing_or_invalid", 'Dict.json is missing or unreadable', source="oxt_check_output/dict.json")
        assets = check_assets(folder, meta, labels, metrics.get("sensors", set()), c)
        declared = {v for key in ("sensor_image", "sensor_array", "sensor_state") for v in strings(meta.get(key))}
        if metrics.get("tables") and declared != metrics["sensors"]:
            c.add("12", "fail", "sensor_names", 'Sensor names differ between metadata and dict', evidence={"metadata_only": sorted(declared - metrics["sensors"]), "dict_only": sorted(metrics["sensors"] - declared)})
        check_supplementary(folder, metrics, labels, c)
        qc = check_existing_qc(folder, c)
        if "ego" in labels and metrics.get("tables"):
            c.applicable.add("14")
            if metrics.get("ego_pose_missing") or metrics.get("ego_rgb_missing"):
                c.add("14", "warning", "ego_fields", 'The ego label is present but camera_ego_rgb/pose is missing', evidence={"tables": metrics["tables"], "rgb_missing": metrics.get("ego_rgb_missing"), "pose_missing": metrics.get("ego_pose_missing")})
        remote = check_remote(meta, options, client, c)
    for name, before in initial_stats.items():
        path = folder / name
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != before:
            c.add("11", "warning", "input_changed", 'An input file changed during the check', source=name)
    summaries = c.summary()
    status = "fail" if any(s["status"] == "fail" for s in summaries) else "needs_review" if any(s["status"] not in ("pass", "not_applicable") for s in summaries) else "pass"
    repository_names = set()
    for url in strings(meta.get("url")):
        try:
            repository_names.add(parse_source(url).repo.rsplit("/", 1)[-1])
        except ValueError:
            continue
    report = {"tool": "precheck_final", "version": VERSION, "generated_at": timestamp(), "input_dir": str(folder),
              "dataset": folder.name, "metadata_name": meta.get("name"), "repository_names": sorted(repository_names),
              "overall_status": status, "submission_ready": status == "pass", "rules": summaries,
              "findings": sorted(c.findings, key=lambda f: (ORDER[f["rule"]], -STATUS_ORDER[f["status"]], f["code"], f["key"])),
              "finding_status_counts": dict(Counter(f["status"] for f in c.findings)),
              "local_metrics": metrics, "assets": assets, "existing_qc": qc, "remote_repositories": remote, "inputs": sources,
              "execution": {"offline": options.offline, "timeout": options.timeout, "repo_budget": options.repo_budget,
                            "max_pages": options.max_pages, "archive_samples": options.archive_samples},
              "review_scope": "machine_only",
              "limits": list(LIMITS_EN)}
    return json_safe(report)


def atomic_write(path: Path, text: str):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


# Legacy report records that require semantic judgments are outside this scope.
ASSUMED_CORRECT_CODES = {
    "annotation_content", "video_content", "viser_content", "extended_pose",
    "state_spatial_candidate", "raw_numeric", "instruction_content", "pose_convention",
    "terminal_key_missing", "estimated_actuator", "provisional_sensor", "supp_semantics",
    "precheck_manual_notes", "vlm_pending", "vlm_coverage", "qc_provenance",
    "archive_payload", "sensor_category", "coordinate_convention", "only_extended_actions",
    "rgb_alpha", "matrix_vector", "legacy_supp_tactile", "imu_as_joints",
    "full_report_omitted", "terminal_provenance",
    "metadata_name", "submission_name", "review_missing_label", "review_forbidden_label",
    "terminal_expected_type", "terminal_scalar_shape", "canonical_unknown",
}
FAIL_TEXT_CN = {
    "code_trigger_not_empty": 'code_trigger 必须填写空字符串 ""',
    "url_count": "下载链接必须且只能有一个",
    "state_shape": "state 触觉只能是三维 (T,N,D)，空间网格请使用 matrix",
    "floating_grid_type": "浮点空间网格触觉必须使用 matrix 类型",
    "image_not_uint8": IMAGE_DTYPE_GUIDANCE_CN,
    "rgb_schema": "RGB 不是 (T,H,W,3) uint8",
    "rgb_missing": "缺少 RGB 数据",
    "invalid_shape": "shape 含无效维度",
    "tactile_members": "触觉组缺少必要成员",
    "invalid_tactile_type": "触觉 type 无效",
    "scalar_axis_missing": "标量 state 的 shape 不符合 (T,N,1)",
    "terminal_scalar_shape": "标量 state 的 shape 不符合 (T,N,1)",
    "terminal_expected_type": "触觉 type 与终审要求不一致",
    "tactile_area_shape": "触觉 data 与 area 的 T/N 不一致",
    "static_area_shape": "静态触觉索引数量与 data 的 N 不一致",
    "tactile_string_shape": "sensor/type 不是 (T,) 字符串数组",
    "functional_area_not_consolidated": "同一功能区域重复占用多个通道",
    "timestamps_missing": "缺少 timestamps",
    "frame_mismatch": "数据帧数与 timestamps 不一致",
    "episode_ends_values": "episode_ends 非递增或末边界不等于 T",
    "no_actions": "缺少 joint/pose/hand 动作信息",
    "joint_idx_missing": "缺少手关节索引",
    "joint_idx_shape": "手关节数据与索引的 shape 不一致",
    "static_idx_shape": "静态动作索引数量与通道数不一致",
    "legacy_static_idx_conflict": "新旧手关节索引不一致",
    "legacy_static_area_conflict": "新旧触觉索引不一致",
    "idx_dtype": "索引不是整数类型",
    "idx_range": "索引超出允许范围",
    "idx_duplicate": "索引重复",
    "constant_changed": "固定索引、名称或类型随时间改变",
    "gripper_area": "夹爪 area 不是 0 或 [0,1]",
    "standard_pose_shape": "标准 pose 不是六维",
    "viser_missing_or_invalid": "缺少有效的 Viser 4D 录像",
    "institution_missing": "缺少 institution 图",
    "sensor_missing": "缺少 sensor 图",
    "image_invalid": "图片不可读取",
    "image_filename": "图片文件名含空格",
    "supp_data_without_label": "有 supplementary 数据但没有 with-supp 标签",
    "video_missing_or_invalid": "缺少有效的 video.mp4",
    "video_square": "video.mp4 不是正方形",
    "instruction_missing": "缺少 sub_task_instruction",
    "instruction_schema": "sub_task_instruction 不是 (T,) 字符串数组",
    "instruction_empty": "sub_task_instruction 为空",
    "metadata_unreadable": "metadata.json 缺失或无法读取",
    "missing_fields": "metadata 缺少必填字段",
    "unfilled": "metadata 字段未填写或不是字符串",
    "field_type": "metadata 字段类型错误",
    "embodiment_type": "metadata.type 不是 gripper/dexhand/human",
    "count_type": "metadata 统计字段不是正整数",
    "frame_count": "metadata.frame 与导出帧数不一致",
    "trajectory_count": "metadata.trajectory 与轨迹数不一致",
    "labels": "含 ego/umi/with-supp 以外的标签",
    "metadata_chinese": "metadata 含中文",
    "contact_email": "contact 不是纯邮箱",
    "reference_type": "reference 不是对象",
    "url_empty": "缺少下载入口",
    "metadata_name": "metadata.name 与表格名称不一致",
    "submission_name": "提交文件夹名与表格名称不一致",
    "review_missing_label": "缺少终审要求的标签",
    "review_forbidden_label": "仍有终审要求删除的标签",
    "dataset_url": "下载入口不是有效的 HF/MS 数据集链接",
    "repo_suffix": "repo 名必须以 _OXT 结尾",
    "partial_entry": "下载入口指向子目录或单个文件",
    "multiple_entries": "同一平台有多个下载入口",
    "readme_missing": "repo 根目录缺少 README.md",
    "dataset_dir_missing": "repo 缺少与仓库名称一致的数据目录",
    "empty_dataset": "数据集目录为空",
    "empty_task": "任务文件夹为空",
    "mixed_layout": "数据目录混放文件和任务文件夹",
    "nested_task": "任务文件夹内还有子目录",
    "non_archive": "数据目录中有非压缩文件",
    "lfs_pointer": "数据文件是 LFS 指针",
    "empty_archive": "数据文件大小为零",
    "qc_dir_missing": "缺少 oxt_check_output 文件夹",
    "dict_missing_or_invalid": "dict.json 缺失或无法读取",
    "qc_result_missing": "缺少 QC 结果文件",
    "precheck_unreadable": "预检结果无法读取",
    "vlm_unreadable": "VLM 结果无法读取",
    "vlm_not_run": "VLM 未实际运行",
    "vlm_score_invalid": "VLM 分数不在 [0,1] 范围",
    "vlm_bucket": "VLM 分数与 pass/warning/fail 分类不一致",
    "vlm_average": "VLM 平均分计算不一致",
    "sensor_names": "metadata 与 dict 中的传感器名称不一致",
    "note_json": "supplementary 说明文件无法读取",
    "note_object": "supplementary 说明不是 JSON 对象",
    "note_dimension": "supplementary dimension 不是正整数",
    "note_dimensions_incomplete": "supplementary 没有完整的维度说明",
    "note_shape_inconsistent": "supplementary shape 与传感器数量、维度不一致",
    "note_placeholder": "supplementary 说明仍含未填写的模板内容",
    "unexplained_joint_dimensions": "supplementary 部分关节维度没有说明",
    "internal_error": "检查脚本执行失败",
}
FAIL_TEXT = {
    "code_trigger_not_empty": 'code_trigger must be an empty string ("")',
    "url_count": "Exactly one download URL is required",
    "state_shape": "State tactile data must be three-dimensional (T,N,D); use matrix for spatial grids",
    "floating_grid_type": "Floating-point spatial tactile data must use type=matrix",
    "image_not_uint8": IMAGE_DTYPE_GUIDANCE,
    "rgb_schema": "RGB must have shape (T,H,W,3) and dtype uint8",
    "rgb_missing": "RGB data is missing",
    "invalid_shape": "Shape contains invalid dimensions",
    "tactile_members": "Required tactile group members are missing",
    "invalid_tactile_type": "Tactile type is invalid",
    "scalar_axis_missing": "Scalar state shape does not match (T,N,1)",
    "terminal_scalar_shape": "Scalar state shape does not match (T,N,1)",
    "terminal_expected_type": "Tactile type does not match the final-review requirement",
    "tactile_area_shape": "Tactile data and area have different T/N dimensions",
    "static_area_shape": "Static tactile index count does not match data dimension N",
    "tactile_string_shape": "Sensor/type must be a (T,) string array",
    "functional_area_not_consolidated": "One functional area occupies duplicate channels",
    "timestamps_missing": "Timestamps are missing",
    "frame_mismatch": "Data frame count does not match timestamps",
    "episode_ends_values": "Episode ends are not increasing or the final boundary is not T",
    "no_actions": "Joint/pose/hand action information is missing",
    "joint_idx_missing": "Hand-joint indices are missing",
    "joint_idx_shape": "Hand-joint data and index shapes do not match",
    "static_idx_shape": "Static action index count does not match the channel count",
    "legacy_static_idx_conflict": "Legacy and static hand-joint indices disagree",
    "legacy_static_area_conflict": "Legacy and static tactile indices disagree",
    "idx_dtype": "Indices must have an integer dtype",
    "idx_range": "Indices are outside the permitted range",
    "idx_duplicate": "Indices contain duplicates",
    "constant_changed": "Fixed indices, names, or types change over time",
    "gripper_area": "Gripper area must be 0 or [0,1]",
    "standard_pose_shape": "Standard pose must have six dimensions",
    "viser_missing_or_invalid": "A valid Viser 4D recording is missing",
    "institution_missing": "An institution image is missing",
    "sensor_missing": "A sensor image is missing",
    "image_invalid": "An image cannot be decoded",
    "image_filename": "An image filename contains spaces",
    "supp_data_without_label": "Supplementary data exists without the with-supp label",
    "video_missing_or_invalid": "A valid video.mp4 is missing",
    "video_square": "The video.mp4 file is not square",
    "instruction_missing": "The sub_task_instruction field is missing",
    "instruction_schema": "The sub_task_instruction field must be a (T,) string array",
    "instruction_empty": "The sub_task_instruction field is empty",
    "metadata_unreadable": "The metadata.json file is missing or unreadable",
    "missing_fields": "Required metadata fields are missing",
    "unfilled": "Metadata fields are empty or are not strings",
    "field_type": "Metadata field types are invalid",
    "embodiment_type": "The metadata.type field must be gripper/dexhand/human",
    "count_type": "Metadata counts must be positive integers",
    "frame_count": "The metadata.frame value does not match the exported frame count",
    "trajectory_count": "The metadata.trajectory value does not match the episode count",
    "labels": "Labels other than ego/umi/with-supp are present",
    "metadata_chinese": "Metadata contains Chinese text",
    "contact_email": "Contact must contain email addresses only",
    "reference_type": "Reference must be an object",
    "url_empty": "A download URL is missing",
    "metadata_name": "The metadata.name field does not match the registered name",
    "submission_name": "The submission folder name does not match the registered name",
    "review_missing_label": "A label required by the final review is missing",
    "review_forbidden_label": "A label rejected by the final review is still present",
    "dataset_url": "The download URL is not a valid HF/MS dataset URL",
    "repo_suffix": "Repository name must end with _OXT",
    "partial_entry": "The download URL points to a subdirectory or a single file",
    "multiple_entries": "One platform has multiple download URLs",
    "readme_missing": "README.md is missing from the repository root",
    "dataset_dir_missing": "The repository lacks a data directory matching its repository name",
    "empty_dataset": "The dataset directory is empty",
    "empty_task": "A task directory is empty",
    "mixed_layout": "The dataset directory mixes files and task directories",
    "nested_task": "A task directory contains nested directories",
    "non_archive": "The data directory contains non-archive files",
    "lfs_pointer": "A data file is an LFS pointer",
    "empty_archive": "A data file has zero size",
    "qc_dir_missing": "The oxt_check_output directory is missing",
    "dict_missing_or_invalid": "The dict.json file is missing or unreadable",
    "qc_result_missing": "Required QC result files are missing",
    "precheck_unreadable": "The precheck result is unreadable",
    "vlm_unreadable": "The VLM result is unreadable",
    "vlm_not_run": "VLM was not actually run",
    "vlm_score_invalid": "VLM scores must be within [0,1]",
    "vlm_bucket": "VLM scores do not match their pass/warning/fail buckets",
    "vlm_average": "The VLM average score is inconsistent",
    "sensor_names": "Sensor names differ between metadata and dict",
    "note_json": "A supplementary note is unreadable",
    "note_object": "A supplementary note must be a JSON object",
    "note_dimension": "Supplementary dimension must be a positive integer",
    "note_dimensions_incomplete": "Supplementary dimension descriptions are incomplete",
    "note_shape_inconsistent": "Supplementary shape disagrees with sensor count and dimensions",
    "note_placeholder": "A supplementary note still contains template placeholders",
    "unexplained_joint_dimensions": "Some supplementary joint dimensions have no description",
    "internal_error": "The check script failed to complete",
}
OTHER_MESSAGES = {
    "gripper_area_unrecorded": ("Gripper local area constants are unavailable or cannot be resolved", "gripper 的局部区域常量缺失或无法解析"),
    "annotation_missing": ("A required annotation image is missing", "缺少必要的标注图"),
    "required_note_missing": ("A required supplementary JSON note is missing", "缺少必要的 supplementary JSON 说明"),
    "tree_partial": ("The dataset directory listing is incomplete", "数据集目录列表不完整"),
    "task_unlisted": ("A task directory has not been fully listed", "任务目录未完整读取"),
    "task_partial": ("A task directory listing is incomplete", "任务目录列表不完整"),
    "archive_unrecognized": ("Archive format could not be identified; formats are unrestricted", "压缩格式不限，但当前格式未识别"),
    "size_unknown": ("File size is unavailable", "文件大小未提供"),
    "canonical_unknown": ("The registered DatasetName could not be uniquely identified", "无法唯一确定表格正式 DatasetName"),
    "pillow_missing": ("Pillow is unavailable; image decoding could not be checked", "Pillow 不可用，未检查图片解码"),
    "sensor_photo_name": ("A sensor has no PNG with the exact matching name", "传感器没有完全同名的 PNG"),
    "video_unverified": ("Video format could not be verified", "视频格式未确认"),
    "viser_unverified": ("Viser recording format could not be verified", "Viser 录像格式未确认"),
    "episode_ends_missing": ("Usable episode_ends are unavailable", "缺少可用的 episode_ends"),
    "type_unrecorded": ("The tactile type has no readable constant value", "触觉 type 没有可读的常量值"),
    "no_tables": ("No data tables could be parsed from dict", "dict 中没有可解析的数据表"),
    "statistics_scope": ("A flat or missing schema cannot establish complete episode statistics", "平铺或缺失 schema 无法证明完整轨迹统计"),
    "compression_unrecorded": ("Compression metadata is missing for some arrays", "部分数组缺少 compression 记录"),
    "image_uncompressed": ("Exported image compression is disabled", "导出的图像压缩配置为空"),
    "supp_file_label": ("Supplementary files and the with-supp label disagree", "supplementary 文件与 with-supp 标签不一致"),
    "supp_label_without_data": ("The with-supp label exists but no supplementary arrays were exported", "有 with-supp 标签，但未导出补充数组"),
    "supp_area_tool_conflict": ("Legacy QC requires supplementary area although the guide permits three members", "旧 QC 要求 supplementary area，但指南允许三件套"),
    "precheck_reported_fail": ("An existing precheck result records a fail", "已有预检结果记录 fail"),
    "precheck_warning": ("An existing precheck result records a warning", "已有预检结果记录 warning"),
    "vlm_score_missing": ("A non-error VLM page has no valid numeric score", "非 error 的 VLM 页面缺少有效评分"),
    "vlm_reported_warning": ("An existing VLM result records a warning", "已有 VLM 结果记录 warning"),
    "vlm_reported_fail": ("An existing VLM result records a fail", "已有 VLM 结果记录 fail"),
    "vlm_reported_error": ("An existing VLM result records an execution error", "已有 VLM 结果记录执行错误"),
    "vlm_no_reports": ("The VLM result has no parseable review records", "VLM 结果没有可解析的审查记录"),
    "vlm_aspect_schema": ("The VLM result has no parseable four-aspect page structure", "VLM 结果缺少可解析的四方面页面结构"),
    "no_remote_entry": ("No readable remote dataset URL is available", "没有可读取的远端数据集入口"),
    "offline": ("Remote checks were skipped in offline mode", "离线模式跳过了远端检查"),
    "remote_unreachable": ("The remote directory could not be read", "远端目录无法读取"),
    "hf_and_mirror_unreachable": ("Neither HF nor the mirror completed the directory read", "HF 和镜像均未完成目录读取"),
    "mirror_used": ("The same repository and revision were checked through hf-mirror", "已通过 hf-mirror 检查相同仓库和版本"),
    "mirror_sync_unverified": ("The mirror was read, but its synchronization with the official branch is unverified", "已读取镜像，但与官方分支的同步状态未确认"),
    "root_partial": ("The repository root listing is incomplete", "仓库根目录列表不完整"),
    "mirror_file_used": ("HF file access failed; the mirror served the same revision and path", "HF 文件访问失败后使用了相同版本和路径的镜像"),
    "readme_oxt": ("No Open-X-Tactile reference was found in the bounded README text", "读取的 README 文本未发现 Open-X-Tactile 引用"),
    "readme_unverified": ("README content could not be read", "README 内容无法读取"),
    "mirror_subtree_used": ("HF subtree access failed; the mirror served the same revision and path", "HF 子目录访问失败后使用了相同版本和路径的镜像"),
    "task_limit": ("The number of task directories exceeds the check limit", "任务目录数超过检查上限"),
    "archive_sample_unavailable": ("The archive header sample could not be read", "压缩包头样本无法读取"),
    "task_folder_count": ("The task directory count differs from metadata.task", "任务目录数与 metadata.task 不一致"),
    "dataset_tree_unavailable": ("The dataset directory was not fully read", "数据集目录未完整读取"),
    "metadata_unavailable": ("Metadata is unavailable; related checks are incomplete", "metadata 不可用，相关检查未完成"),
    "dict_unavailable": ("The dict.json file is unavailable or cannot be parsed", "dict.json 不可用或无法解析"),
    "ego_fields": ("The ego label is present but camera_ego_rgb/pose is missing", "有 ego 标签，但缺少 camera_ego_rgb/pose"),
    "input_changed": ("An input file changed during the check", "检查期间输入文件发生改变"),
}
MESSAGES_EN = {**FAIL_TEXT, **{code: text[0] for code, text in OTHER_MESSAGES.items()}}
MESSAGES_CN = {**FAIL_TEXT_CN, **{code: text[1] for code, text in OTHER_MESSAGES.items()}}
LIMITS_EN = ["Only metadata, dict, attachments, and bounded remote directory information are checked.",
             "Unknown means network access or exported information was insufficient to complete a machine check."]
LIMITS_CN = ["只检查 metadata、dict、附件和有界远端目录信息。",
             "unknown 表示网络访问或导出信息不足，相关机器检查未完成。"]
DIAGNOSTIC_TEXT = {
    "网络访问失败：": "Network access failed: ",
    "表格读取失败：": "Workbook read failed: ",
    "正式名称重复：": "Duplicate registered name: ",
    "未找到 Contributor List；无法证明正式 DatasetName": "Contributor List was not found; the registered DatasetName is unavailable",
    "指定名称不在表格中": "The specified name is not registered in the workbook",
    "无法唯一定位正式表格名称": "The registered name could not be uniquely identified",
    "仅用于找到表格行；正式名称检查仍严格比较原字符串": "Used only to identify a workbook row; compliance still requires the exact registered spelling",
    "仓库检查达到时间预算": "The repository time budget was exhausted",
    "达到目录项/分页上限": "The entry/page limit was reached",
    "官方/镜像目录均未能完成读取": "Neither the official endpoint nor the mirror completed the listing",
    "MS 目录未能读取": "The MS directory could not be read",
    "API 响应超过有界读取上限，列表完整性未知": "The API response exceeded the bounded read limit; listing completeness is unknown",
    "API 未返回可解析的 JSON": "The API did not return parseable JSON",
    "目录项不是对象": "A directory entry is not an object",
    "目录项 path 缺失或不合法": "A directory entry has a missing or invalid path",
    "目录接口返回了所请求目录之外的项": "The API returned an entry outside the requested directory",
    "非递归目录返回深层内容，直接子项覆盖不明确": "A non-recursive listing returned deep entries; direct-child coverage is unclear",
    "未知目录项类型": "Unknown directory entry type",
    "HF 分页循环": "HF pagination loop detected",
    "HF tree 返回格式不正确": "Invalid HF tree response format",
    "HF 分页重复目录项，快照可能改变": "HF pagination repeated an entry; the snapshot may have changed",
    "HF 分页域名改变": "The HF pagination hostname changed",
    "MS tree API 未返回成功的 Data": "The MS tree API returned unsuccessful Data",
    "MS tree 缺少 Files 列表": "The MS tree response has no Files list",
    "MS 分页重复目录项，未确认列表完整": "MS pagination repeated an entry; completeness is unknown",
    "MS 分页 TotalCount 改变": "MS pagination TotalCount changed",
    "MS 返回短页但未达到 TotalCount": "MS returned a short page before TotalCount was reached",
    "镜像 revision 与已知官方 commit 不一致": "The mirror revision differs from the known official commit",
    "文件缺失": "File is missing",
    "ffprobe 不可用": "ffprobe is unavailable",
    "无有效视频流": "No valid video stream",
    "视频探测失败：": "Video probing failed: ",
}


def localize_diagnostics(value: Any) -> Any:
    """Translate authored legacy diagnostics while retaining source identifiers."""
    if isinstance(value, list):
        return [localize_diagnostics(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if isinstance(item, (list, dict)):
            item = localize_diagnostics(item)
        elif key in ("error", "reason", "unknown", "note") and isinstance(item, str):
            for original, translated in DIAGNOSTIC_TEXT.items():
                item = item.replace(original, translated)
        result[key] = item
    return result
KEYED_FAIL_CODES = {
    "state_shape", "floating_grid_type",
    "image_not_uint8", "rgb_schema", "invalid_shape", "tactile_members",
    "invalid_tactile_type", "scalar_axis_missing", "terminal_scalar_shape",
    "terminal_expected_type", "tactile_area_shape", "static_area_shape",
    "tactile_string_shape", "functional_area_not_consolidated", "frame_mismatch",
    "joint_idx_missing", "joint_idx_shape", "static_idx_shape", "idx_dtype",
    "idx_range", "idx_duplicate", "constant_changed", "gripper_area",
    "standard_pose_shape", "unfilled", "field_type", "count_type",
    "review_missing_label", "review_forbidden_label",
}


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().rstrip("。.")


def source_text(value: Any, language: str = "en") -> str:
    """Escape Chinese source characters in English views without changing evidence."""
    text = one_line(value)
    return CJK_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", text) if language == "en" else text


def message_text(code: str, fallback: str = "", language: str = "en") -> str:
    messages = MESSAGES_CN if language == "cn" else MESSAGES_EN
    if code in messages:
        return messages[code]
    if language == "en" and CJK_RE.search(fallback):
        return "Check result: " + code.replace("_", " ")
    return one_line(fallback) or ("检查失败" if language == "cn" else "Check failed")


def without_human_requests(value: Any) -> Any:
    """Keep machine evidence while omitting review requests copied from old QC."""
    if isinstance(value, dict):
        return {key: without_human_requests(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_human_requests(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"[,，;；。]?\s*(?:需(?:要)?(?:进行)?人工|请(?:进行)?人工|建议人工|待人工)(?:核查|核验|核对|复核|检查|确认|查看).*", "", value, flags=re.S)
    return value


def failure_points(report: dict, language: str = "en") -> list[str]:
    """One sentence per distinct fail, grouping repeated files and array keys."""
    groups: dict[str, set[str]] = {}
    for item in report.get("findings", []):
        code = item.get("code", "")
        if item.get("status") != "fail" or code in ASSUMED_CORRECT_CODES:
            continue
        text = message_text(code, item.get("message", ""), language)
        evidence = next((e.get("evidence") for e in item.get("examples", [])
                         if isinstance(e.get("evidence"), dict)), {})
        if code == "repo_suffix":
            text = "repo 名必须以 _OXT 结尾" if language == "cn" else "Repository name must end with _OXT"
        elif code == "dataset_dir_missing" and evidence.get("expected"):
            name = evidence["expected"]
            text = f"repo 缺少同名的 {source_text(name, language)}/ 文件夹" if language == "cn" else f"The repository is missing the matching {source_text(name)}/ directory"
        elif code == "terminal_expected_type" and evidence.get("expected"):
            text = f"触觉 type 应为 {source_text(evidence['expected'], language)}" if language == "cn" else f"Tactile type must be {source_text(evidence['expected'])}"
        elif code == "annotation_missing":
            if language == "cn":
                text = "缺少 " + {"hand_joints_idx.png": "hand joint 图", "tactile_func_area_idx.png": "func_area 图"}.get(item.get("key"), "标注图")
            else:
                text = {"hand_joints_idx.png": "The hand-joint image is missing", "tactile_func_area_idx.png": "The functional-area image is missing"}.get(item.get("key"), "An annotation image is missing")
        elif code in ("qc_result_missing", "required_note_missing"):
            filename = PurePosixPath(str(item.get("source", "")).replace("\\", "/")).name
            text = "缺少 " + source_text(filename or "必要结果文件", language) if language == "cn" else "Missing " + source_text(filename or "required result file")
        keys = groups.setdefault(one_line(text), set())
        if code in KEYED_FAIL_CODES and item.get("key"):
            keys.add(source_text(item["key"], language))
    left, separator, right, period = ("（", "、", "）", "。") if language == "cn" else (" (", ", ", ")", ".")
    return [one_line(text) + (left + separator.join(sorted(keys)) + right if keys else "") + period
            for text, keys in groups.items()]


def prepare_machine_report(report: dict) -> dict:
    """Refresh presentation of saved results without repeating remote requests."""
    c = Collector()
    c.applicable = {r["rule"] for r in report.get("rules", []) if r["status"] != "not_applicable"}
    for original in report.get("findings", []):
        if original.get("status") == "manual" or original.get("code") in ASSUMED_CORRECT_CODES:
            continue
        item = dict(original)
        item["message"] = message_text(item.get("code", ""), item.get("message", ""))
        item["examples"] = without_human_requests(item.get("examples", []))
        c.findings.append(item)
    report["findings"] = c.findings
    report["rules"] = c.summary()
    report["finding_status_counts"] = dict(Counter(f["status"] for f in c.findings))
    report["overall_status"] = "fail" if any(f["status"] == "fail" for f in c.findings) else "needs_review" if any(f["status"] in ("warning", "unknown") for f in c.findings) else "pass"
    report["submission_ready"] = report["overall_status"] == "pass"
    report["review_scope"] = "machine_only"
    report["language"] = "en"
    report["report_format_version"] = VERSION
    report["rendered_at"] = timestamp()
    report["failure_points"] = failure_points(report)
    report["limits"] = list(LIMITS_EN)
    for key in ("remote_repositories", "assets"):
        if key in report:
            report[key] = localize_diagnostics(report[key])
    for key in ("workbook", "canonical_name", "canonical_resolution"):
        report.pop(key, None)
    report.get("existing_qc", {}).pop("provenance_verified", None)
    return report


REPORT_LABELS = {
    "en": {"no_fail": "No fail findings.", "details": "Machine check details", "dataset": "Dataset", "name": "Repository names",
           "status": "Status", "when": "Checked at", "scope": "Check coverage", "issues": "Findings and evidence",
           "no_issues": "No machine check findings were recorded.", "field": "Field", "source": "Source", "count": "Occurrences",
           "evidence": "Samples and evidence", "remote": "Remote checks", "no_remote": "No remote access records are available.",
           "url": "URL", "endpoint": "Endpoint", "pinned": "Pinned revision", "access": "Access and coverage records",
           "limits": "Check limits", "json": "Complete JSON report", "other": "Chinese Markdown report", "missing": "Not provided",
           "unresolved": "Unresolved", "unreachable": "Unreachable", "unverified": "Unverified"},
    "cn": {"no_fail": "未发现 fail。", "details": "机器检查详情", "dataset": "数据集", "name": "仓库名称",
           "status": "状态", "when": "检查时间", "scope": "检查范围", "issues": "问题与证据",
           "no_issues": "没有机器检查问题记录。", "field": "字段", "source": "来源", "count": "出现次数",
           "evidence": "样本与证据", "remote": "远端检查", "no_remote": "没有远端读取记录。",
           "url": "入口", "endpoint": "读取端点", "pinned": "固定版本", "access": "访问与覆盖记录",
           "limits": "检查边界", "json": "完整 JSON 报告", "other": "英文 Markdown 报告", "missing": "未提供",
           "unresolved": "无法确认", "unreachable": "未能访问", "unverified": "未确认"},
}


def report_markdown(report: dict, language: str = "en") -> str:
    if language not in REPORT_LABELS:
        raise ValueError("Unsupported report language: " + language)
    label = REPORT_LABELS[language]
    period = "。" if language == "cn" else "."
    titles = RULES_CN if language == "cn" else RULES
    points = failure_points(report, language)
    lines = [f"{i}. {html.escape(text)}" for i, text in enumerate(points, 1)] if points else [label["no_fail"]]
    esc = lambda value: html.escape(source_text(value, language))
    lines += ["", "---", "", "## " + label["details"], "",
              f"{label['dataset']}: {esc(report.get('dataset', label['missing']))}; {label['name']}: {esc(', '.join(report.get('repository_names', [])) or label['missing'])}; {label['status']}: **{esc(report.get('overall_status', 'unknown'))}**{period}", "",
              f"{label['when']}: {esc(report.get('generated_at', label['missing']))}{period}", "",
              "## " + label["scope"], ""]
    for row in report.get("rules", []):
        title = titles.get(row["rule"], row["title"])
        count = f"{row['finding_groups']} 组记录" if language == "cn" else f"{row['finding_groups']} finding groups"
        lines.append(f"- **{esc(row['rule'])} · {esc(title)}**: {esc(row['status'])}; {count}{period}")
    lines += ["", "## " + label["issues"], ""]
    findings = [f for f in report.get("findings", [])
                if f["status"] != "manual" and f["code"] not in ASSUMED_CORRECT_CODES]
    if not findings:
        lines.append(label["no_issues"])
    for row in report.get("rules", []):
        items = [f for f in findings if f["rule"] == row["rule"]]
        if not items:
            continue
        lines += [f"### {esc(row['rule'])} · {esc(titles.get(row['rule'], row['title']))}", ""]
        for item in items:
            lines += [f"#### {esc(item['status'])} · {esc(item['code'])}", "",
                      esc(message_text(item["code"], item["message"], language)) + period, ""]
            if item.get("key"):
                lines += [f"{label['field']}: {esc(item['key'])}{period}", ""]
            if item.get("source"):
                lines += [f"{label['source']}: {esc(item['source'])}{period}", ""]
            lines += [f"{label['count']}: {item['occurrences']}{period}", ""]
            if item.get("examples"):
                lines += [label["evidence"] + ":", "", "```json",
                          json.dumps(item["examples"], ensure_ascii=language != "cn", indent=2), "```", ""]
    lines += ["", "## " + label["remote"], ""]
    remotes = report.get("remote_repositories", [])
    if not remotes:
        lines += [label["no_remote"], ""]
    for remote in remotes:
        lines += [f"### {esc(remote['provider'])} · {esc(remote['repo'])}", "",
                  f"{label['url']}: {esc(remote.get('url', label['missing']))}{period}", "",
                  f"{label['endpoint']}: {esc(remote.get('endpoint') or label['unreachable'])}; revision: {esc(remote.get('revision') or label['unverified'])}; {label['pinned']}: {remote.get('pinned', False)}{period}", ""]
        coverage = {key: remote[key] for key in ("error", "attempts", "fallback_paths", "official_sync_unverified", "layout", "archive_header_samples") if key in remote}
        for key in ("root", "dataset_tree"):
            if key in remote:
                tree = remote[key]
                coverage[key] = {"complete": tree.get("complete"), "pages": tree.get("pages"),
                                 "entry_count": len(tree.get("entries", [])), "reason": tree.get("reason", "")}
        if "tasks" in remote:
            coverage["tasks"] = {path: {"complete": tree.get("complete"), "pages": tree.get("pages"),
                                       "entry_count": len(tree.get("entries", [])), "reason": tree.get("reason", "")}
                                 for path, tree in remote["tasks"].items()}
        if coverage:
            lines += [label["access"] + ":", "", "```json", json.dumps(coverage, ensure_ascii=language != "cn", indent=2), "```", ""]
    lines += ["## " + label["limits"], ""]
    lines += ["- " + esc(value) + period for value in (LIMITS_CN if language == "cn" else LIMITS_EN)]
    other_file = "precheck_final_report.md" if language == "cn" else "precheck_final_report_cn.md"
    lines += ["", f"[{label['json']}](precheck_final_report.json) · [{label['other']}]({other_file})", ""]
    return "\n".join(lines)


STYLE = """body{font:14px system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 20px;color:#20252a;background:#fff}h1{font-size:24px}h2{font-size:18px;margin-top:26px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd;vertical-align:top}th{background:#f3f4f5;position:sticky;top:0}code,pre{font-family:ui-monospace,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:10px}details{margin:8px 0}summary{cursor:pointer;padding:6px 0;font-weight:600}.fail{color:#aa2929}.unknown,.warning,.needs_review{color:#865817}.pass{color:#236534}.meta{color:#59636e}input,select{font:inherit;padding:6px;margin:8px 10px 12px 0}a{color:#265e9d}.fail-list li{margin:8px 0;overflow-wrap:anywhere}.limits{border-left:3px solid #aaa;padding-left:12px}"""


def report_html(report: dict) -> str:
    esc = lambda value: html.escape(source_text(value))
    points = failure_points(report)
    overview = "<ol class='fail-list'>" + "".join(f"<li>{esc(text)}</li>" for text in points) + "</ol>" if points else "<p>No fail findings.</p>"
    rows = "".join(f"<tr><td>{esc(r['rule'])}</td><td>{esc(r['title'])}</td><td class='{esc(r['status'])}'>{esc(r['status'])}</td><td>{r['finding_groups']}</td></tr>" for r in report["rules"])
    sections = []
    for item in report["findings"]:
        if item["status"] == "manual" or item["code"] in ASSUMED_CORRECT_CODES:
            continue
        example = json.dumps(item["examples"], ensure_ascii=True, indent=2)
        sections.append(f"<details class='finding' data-status='{esc(item['status'])}'><summary><span class='{esc(item['status'])}'>{esc(item['status'])}</span> · {esc(item['rule'])} / {esc(item['code'])} · {esc(item['key'])}</summary><p>{esc(item['message'])} · {item['occurrences']} occurrences</p><p class='meta'>Source: {esc(item['source'])}</p><pre>{esc(example)}</pre></details>")
    limits = "".join(f"<li>{esc(x)}</li>" for x in report["limits"])
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OXT final · {esc(report['dataset'])}</title><style>{STYLE}</style><body>{overview}<p class="meta">{esc(report['dataset'])} · {esc(report['generated_at'])}</p><details open><summary>Machine check details</summary><h2>Check coverage</h2><table><thead><tr><th>Rule</th><th>Check</th><th>Status</th><th>Finding groups</th></tr></thead><tbody>{rows}</tbody></table><h2>Findings and evidence</h2><input id="search" placeholder="Search fields or findings"><select id="status"><option value="">All statuses</option><option>fail</option><option>warning</option><option>unknown</option><option>pass</option></select><div>{''.join(sections)}</div><ul class="limits">{limits}</ul></details><p><a href="precheck_final_report.json">Complete JSON</a> · <a href="precheck_final_report.md">English Markdown</a> · <a href="precheck_final_report_cn.md">Chinese Markdown</a></p><script>function filter(){{let q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('status').value;document.querySelectorAll('.finding').forEach(e=>e.hidden=!(e.textContent.toLowerCase().includes(q)&&(!s||e.dataset.status===s)))}}document.getElementById('search').oninput=filter;document.getElementById('status').onchange=filter;</script></body></html>"""


def save_report(report: dict):
    prepare_machine_report(report)
    folder = Path(report["input_dir"])
    atomic_write(folder / "precheck_final_report.json", json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n")
    atomic_write(folder / "precheck_final_report.md", report_markdown(report))
    atomic_write(folder / "precheck_final_report_cn.md", report_markdown(report, "cn"))
    atomic_write(folder / "precheck_final_report.html", report_html(report))


def save_summary(root: Path, reports: list[dict]):
    for report in reports:
        prepare_machine_report(report)
    rows = [{"dataset": r["dataset"], "repository_names": r.get("repository_names", []), "status": r["overall_status"],
             "finding_status_counts": r["finding_status_counts"], "report": r["dataset"] + "/precheck_final_report.html",
             "rules": {v["rule"]: v["status"] for v in r["rules"]},
             "failure_points": r["failure_points"]} for r in reports]
    summary = {"tool": "precheck_final", "version": VERSION, "generated_at": timestamp(), "input_root": str(root),
               "dataset_count": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
               "review_scope": "machine_only", "datasets": rows}
    atomic_write(root / "precheck_final_summary.json", json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    markdown = ["# OXT machine check summary", "", f"{len(rows)} submissions: {summary['status_counts']}", ""]
    for row in rows:
        brief = " ".join(f"{i}. {text}" for i, text in enumerate(row["failure_points"], 1)) or "No fail findings."
        markdown.append(f"- [{source_text(row['dataset'])}](<{row['report']}>): {html.escape(brief)}")
    atomic_write(root / "precheck_final_summary.md", "\n".join(markdown) + "\n")
    table_rows = []
    for row in rows:
        brief = "<ol class='fail-list'>" + "".join(f"<li>{html.escape(text)}</li>" for text in row["failure_points"]) + "</ol>" if row["failure_points"] else "No fail findings."
        table_rows.append(f"<tr class='dataset-row'><td><a href='{html.escape(row['report'], quote=True)}'>{html.escape(source_text(row['dataset']))}</a></td><td class='{row['status']}'>{row['status']}</td><td>{brief}</td></tr>")
    page = f"<!doctype html><html lang='en'><meta charset='utf-8'><title>OXT final batch</title><style>{STYLE}</style><h1>OXT machine check summary</h1><p>{len(rows)} submissions · {html.escape(str(summary['status_counts']))}</p><input id='q' placeholder='Search datasets or findings'><table><thead><tr><th>Dataset / report</th><th>Status</th><th>Fail findings</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table><script>document.getElementById('q').oninput=function(){{let q=this.value.toLowerCase();document.querySelectorAll('.dataset-row').forEach(e=>e.hidden=!e.textContent.toLowerCase().includes(q))}}</script></html>"
    atomic_write(root / "precheck_final_summary.html", page)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Final metadata folder, or its parent when using --batch")
    parser.add_argument("--batch", action="store_true", help="Check all non-hidden submission folders under the parent")
    parser.add_argument("--offline", action="store_true", help="Skip network access and mark remote checks unknown")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent submission workers; default: 4")
    parser.add_argument("--timeout", type=float, default=8.0, help="Timeout per HTTP request, in seconds")
    parser.add_argument("--repo-budget", type=float, default=40.0, help="Time budget per repository root/content phase, in seconds")
    parser.add_argument("--max-pages", type=int, default=12)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--max-entries", type=int, default=20000)
    parser.add_argument("--archive-samples", type=int, default=1, help="Archive header samples per dataset; unknown formats also require identification")
    args = parser.parse_args(argv)
    root = args.input.resolve()
    if not root.is_dir():
        parser.error("Input must be a directory")
    if not 1 <= args.workers <= 16 or args.timeout <= 0 or args.repo_budget <= 0 or args.max_pages < 1 or args.max_tasks < 1 or args.max_entries < 1 or args.archive_samples < 0:
        parser.error("Invalid workers, timeout, or limit arguments")
    options = Options(offline=args.offline,
                      timeout=args.timeout, repo_budget=args.repo_budget, max_pages=args.max_pages, max_tasks=args.max_tasks,
                      max_entries=args.max_entries, archive_samples=args.archive_samples)
    client = RemoteClient(options)
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if args.batch else [root]
    if not folders:
        parser.error("No submission folders were found")

    def process(folder):
        try:
            report = review_dataset(folder, options, client)
        except Exception as error:
            # Keep the batch running, but never turn internal errors into pass.
            report = {"tool": "precheck_final", "version": VERSION, "generated_at": timestamp(), "input_dir": str(folder),
                      "dataset": folder.name, "repository_names": [], "overall_status": "fail", "submission_ready": False,
                      "rules": [{"rule": key, "title": title, "status": "unknown", "finding_groups": 1} for key, title in RULES.items()],
                      "findings": [{"rule": "11", "status": "fail", "code": "internal_error", "message": "The check did not complete; see the error evidence",
                                    "key": "", "source": "", "occurrences": 1, "examples": [{"evidence": f"{type(error).__name__}: {error}"}]}],
                      "finding_status_counts": {"fail": 1}, "remote_repositories": [], "limits": ["Internal error; all unfinished checks are unknown"]}
        save_report(report)
        return report

    reports = []
    with ThreadPoolExecutor(max_workers=args.workers if args.batch else 1) as pool:
        futures = {pool.submit(process, folder): folder for folder in folders}
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            counts = report["finding_status_counts"]
            print(f"[{len(reports)}/{len(folders)}] {report['dataset']}: {report['overall_status']} ({counts})", flush=True)
    reports.sort(key=lambda r: r["dataset"])
    if args.batch:
        summary = save_summary(root, reports)
        print(json.dumps({"summary": str(root / "precheck_final_summary.html"), "dataset_count": len(reports), "status_counts": summary["status_counts"]}, ensure_ascii=True))
    else:
        print(str(root / "precheck_final_report.html"))
    return 1 if any(r["overall_status"] == "fail" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
