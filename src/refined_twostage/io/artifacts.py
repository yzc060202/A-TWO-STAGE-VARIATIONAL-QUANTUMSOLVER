"""Artifact, config, and report helpers."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Mapping

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def md_table(rows: Iterable[Mapping[str, Any]], fields: List[str]) -> str:
    rows = list(rows)
    if not fields:
        return ""
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        vals = []
        for key in fields:
            val = row.get(key, "")
            vals.append("{:.6e}".format(val) if isinstance(val, float) else str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def environment_record() -> dict:
    record = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "cwd": os.getcwd(), "time_unix": time.time()}
    for name in ("numpy", "scipy", "yaml"):
        try:
            module = __import__(name)
            record[name + "_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            record[name + "_version"] = "unavailable: {}".format(exc)
    try:
        record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root().parent), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        record["git_commit"] = "unavailable"
    return record


def timestamped_result_root(base: Path, label: str = "standard_framework_build") -> Path:
    root = base / "{}_{}".format(label, time.strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True, exist_ok=False)
    (base / "LATEST_RESULT_ROOT.txt").write_text(str(root.resolve()), encoding="utf-8")
    return root


def latest_result_root(base: Path) -> Path:
    latest = base / "LATEST_RESULT_ROOT.txt"
    if latest.exists():
        path = Path(latest.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    roots = sorted(base.glob("standard_framework_build_*"))
    if not roots:
        raise FileNotFoundError("no standard_framework_build_* result root found under {}".format(base))
    return roots[-1]
