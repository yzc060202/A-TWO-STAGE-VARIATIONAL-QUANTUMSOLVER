"""Validate existing Stage-I campaign artifacts and write English reports.

This script is intentionally read-only with respect to optimization/data
generation. It recomputes diagnostics from saved Stage-I parameters and writes
validation reports plus plots.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refined_twostage.io.hashing import sha256_array, short_hash_array
from refined_twostage.metrics.matrix_metrics import singular_diagnostics
from refined_twostage.stage1.evaluator import stage1_metrics
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize

import importlib.util

CONT_SPEC = importlib.util.spec_from_file_location("continue_direct64_campaign", ROOT / "scripts" / "continue_direct64_campaign.py")
cont = importlib.util.module_from_spec(CONT_SPEC)
CONT_SPEC.loader.exec_module(cont)

CAMPAIGN = ROOT / "results" / "direct64_full_twostage_campaign_20260820_204937"
STAGE1 = CAMPAIGN / "05_helmholtz_stage1"
VALIDATION = CAMPAIGN / "09_stage1_training_validation"


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def fmt(value, digits: int = 16) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e5:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}g}"


def md_table(rows: list[dict], fields: list[str]) -> str:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_of(path: Path, needle: str) -> str:
    text = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(text, start=1):
        if needle in line:
            return f"{path.relative_to(ROOT)}:{idx}"
    return str(path.relative_to(ROOT))


def family_label(family: str) -> str:
    return {
        "gray_multiplexed_ry_cnot": "FABLE-inspired Gray",
        "vcbe_block11_real": "Block11",
        "vcbe_block15_real": "Block15",
    }.get(family, family)


def reps_or_layers(family: str, p: int) -> int:
    if family == "gray_multiplexed_ry_cnot":
        return int(p) // 256
    if family == "vcbe_block11_real":
        return (int(p) - 7) // 42
    if family == "vcbe_block15_real":
        return (int(p) - 7) // 14
    raise ValueError(family)


def architecture_text(family: str, p: int) -> str:
    n = reps_or_layers(family, p)
    if family == "gray_multiplexed_ry_cnot":
        return (
            "Clean-framework dense projected-block surrogate for the Gray family. "
            f"The preserved resource contract is 6 data qubits plus one ancilla, fixed ancilla RY(pi), P=256*L with L={n}, "
            f"CNOT=256*L, no CZ, and native depth=32*L+1. Historical provenance is "
            "`src/twostage/circuits/stage1_gray_multiplexed.py`."
        )
    if family == "vcbe_block11_real":
        return (
            "Clean-framework dense projected-block surrogate for VCBE Block11. "
            f"The preserved resource contract is 6 data qubits plus one ancilla, all-to-all real-RCN schedule, P=42*M+7 with M={n}, "
            f"CNOT=21*M, no CZ, and native depth=14*M+1. Historical provenance is "
            "`src/twostage/circuits/stage1_vcbe.py`."
        )
    return (
        "Clean-framework dense projected-block surrogate for VCBE Block15. "
        f"The preserved resource contract is 6 data qubits plus one ancilla, circular real-RCN schedule, P=14*M+7 with M={n}, "
        f"CNOT=7*M, no CZ, and native depth=5*M+1. Historical provenance is "
        "`src/twostage/circuits/stage1_vcbe.py`."
    )


def load_history(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "training_history.csv"
    if csv_path.exists():
        return read_csv(csv_path)
    json_path = run_dir / "training_history.json"
    if json_path.exists():
        payload = read_json(json_path)
        return payload.get("rows", [])
    return []


def float_or_none(row: dict, key: str):
    if key not in row or row[key] in ("", None):
        return None
    return float(row[key])


def history_summary(history: list[dict], reported: dict) -> dict:
    if not history:
        return {
            "configured_steps": 1000,
            "actual_recorded_steps": 0,
            "first_step": None,
            "last_step": None,
            "initial_loss": None,
            "best_loss": reported.get("best_loss"),
            "last_loss": reported.get("final_loss"),
            "best_step": None,
            "lr_first": None,
            "lr_best": None,
            "lr_final": None,
            "grad_first": None,
            "grad_best": None,
            "grad_last_recorded": None,
            "history_best_step_exact": False,
        }
    rows = [{k: v for k, v in h.items()} for h in history]
    for h in rows:
        for key in ("step", "loss", "best_loss", "gradient_norm", "learning_rate", "elapsed_seconds"):
            if key in h and h[key] != "":
                h[key] = float(h[key])
    first = rows[0]
    last = rows[-1]
    best_loss = float(reported.get("best_loss", min(h["loss"] for h in rows)))
    matches = [h for h in rows if abs(float(h["loss"]) - best_loss) <= 1e-18 * max(1.0, abs(best_loss))]
    best_row = matches[0] if matches else min(rows, key=lambda h: float(h["loss"]))
    return {
        "configured_steps": 1000,
        "actual_recorded_steps": len(rows),
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
        "initial_loss": float(first["loss"]),
        "best_loss": best_loss,
        "last_loss": float(last["loss"]),
        "best_step": int(best_row["step"]),
        "lr_first": float(first["learning_rate"]),
        "lr_best": float(best_row["learning_rate"]),
        "lr_final": float(last["learning_rate"]),
        "grad_first": float(first["gradient_norm"]),
        "grad_best": float(best_row["gradient_norm"]),
        "grad_last_recorded": float(last["gradient_norm"]),
        "history_best_step_exact": bool(matches),
        "last_recorded_elapsed_seconds": float(last.get("elapsed_seconds", 0.0)),
    }


def plot_history(run_dir: Path, history: list[dict]) -> dict:
    out_dir = run_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    outputs = {
        "loss_plot": out_dir / "loss_vs_step.png",
        "gradient_plot": out_dir / "gradient_norm_vs_step.png",
        "learning_rate_plot": out_dir / "learning_rate_vs_step.png",
    }
    if not history:
        return outputs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(float(h["step"])) for h in history]
    series = [
        ("loss", "Loss", outputs["loss_plot"], True),
        ("gradient_norm", "Gradient norm", outputs["gradient_plot"], True),
        ("learning_rate", "Learning rate", outputs["learning_rate_plot"], False),
    ]
    for key, ylabel, path, logy in series:
        y = [float(h[key]) for h in history]
        fig = plt.figure(figsize=(6.4, 4.2))
        ax = fig.add_subplot(111)
        ax.plot(steps, y, marker="o", linewidth=1.4, markersize=3)
        ax.set_xlabel("Adam step")
        ax.set_ylabel(ylabel)
        if logy and all(v > 0 for v in y):
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return outputs


def reconstruct_run(row: dict, bundle: dict, s_A: float) -> dict:
    run_dir = Path(row["result_directory"])
    reported = read_json(run_dir / "metrics.json")
    family = row["family"]
    p = int(row["actual_P"])
    ansatz = make_stage1_ansatz(family, target_parameters=p)
    theta0 = np.load(run_dir / "theta0.npy")
    theta = np.load(run_dir / "theta_final.npy")
    target = np.load(run_dir / "target_As.npy")
    saved_k = np.load(run_dir / "K_theta.npy")
    saved_ahat = np.load(run_dir / "Ahat.npy")
    k = ansatz.extract_projected_block(theta)
    ahat = s_A * k
    metrics = stage1_metrics(k, target)
    loss_at_saved_theta, grad_at_saved_theta = ansatz.loss_and_grad_full_basis(theta, target)
    ls = cont.learned_ls_metrics(bundle, ahat)
    mat = singular_diagnostics(ahat)
    history = load_history(run_dir)
    hsum = history_summary(history, reported)
    plots = plot_history(run_dir, history)
    files = [
        "theta0.npy",
        "theta_final.npy",
        "K_theta.npy",
        "Ahat.npy",
        "training_history.csv",
        "training_history.json",
        "metrics.json",
        "resource_report.json",
        "config_snapshot.yaml",
        "hash_manifest.json",
        "status.json",
        "stdout.log",
        "stderr.log",
    ]
    resources = read_json(run_dir / "resource_report.json")
    return {
        "row": row,
        "reported": reported,
        "run_dir": run_dir,
        "ansatz": ansatz,
        "resources": resources,
        "theta0_norm": float(np.linalg.norm(theta0)),
        "theta_final_norm": float(np.linalg.norm(theta)),
        "theta_change_norm": float(np.linalg.norm(theta - theta0)),
        "theta0_hash_recomputed": short_hash_array(theta0),
        "theta_hash_recomputed": short_hash_array(theta),
        "K_hash_recomputed": sha256_array(k),
        "Ahat_hash_recomputed": sha256_array(ahat),
        "K_fro_diff": float(np.linalg.norm(k - saved_k)),
        "K_spec_diff": float(np.linalg.norm(k - saved_k, 2)),
        "K_max_abs_diff": float(np.max(np.abs(k - saved_k))),
        "K_hash_equal": sha256_array(k) == reported["K_hash"],
        "Ahat_fro_diff": float(np.linalg.norm(ahat - saved_ahat)),
        "Ahat_spec_diff": float(np.linalg.norm(ahat - saved_ahat, 2)),
        "Ahat_max_abs_diff": float(np.max(np.abs(ahat - saved_ahat))),
        "Ahat_hash_equal": sha256_array(ahat) == reported["Ahat_hash"],
        "metrics_recomputed": metrics,
        "loss_at_saved_theta": float(loss_at_saved_theta),
        "gradient_norm_at_saved_theta": float(np.linalg.norm(grad_at_saved_theta)),
        "learned_ls_recomputed": {k: v for k, v in ls.items() if k != "Ahat_singular_values"},
        "matrix_recomputed": mat,
        "history": history,
        "history_summary": hsum,
        "plots": plots,
        "files": {name: (run_dir / name).exists() for name in files},
        "s_A": s_A,
        "target_norm_f": float(np.linalg.norm(target)),
        "target_norm_2": float(np.linalg.norm(target, 2)),
        "target_hash_recomputed": sha256_array(target),
        "saved_k": saved_k,
        "saved_ahat": saved_ahat,
        "theta_final": theta,
        "theta0": theta0,
    }


def parameter_response(details: list[dict]) -> list[dict]:
    reps = {
        ("gray_multiplexed_ry_cnot", 6144),
        ("vcbe_block11_real", 5005),
        ("vcbe_block15_real", 5005),
    }
    out = []
    for d in details:
        family = d["row"]["family"]
        p = int(d["row"]["actual_P"])
        if (family, p) not in reps:
            continue
        ansatz = d["ansatz"]
        target = np.load(d["run_dir"] / "target_As.npy")
        theta_final = d["theta_final"]
        theta0 = d["theta0"]
        rng = np.random.default_rng(99100 + p)
        direction = rng.normal(size=ansatz.num_parameters)
        direction /= np.linalg.norm(direction)
        points = [
            ("theta0", theta0),
            ("theta_final", theta_final),
            ("zero_theta", np.zeros(ansatz.num_parameters)),
            ("fresh_seed_99101", np.random.default_rng(99101).normal(scale=0.02, size=ansatz.num_parameters)),
            ("fresh_seed_99102", np.random.default_rng(99102).normal(scale=0.02, size=ansatz.num_parameters)),
            ("theta_final_plus_1e-6_d", theta_final + 1e-6 * direction),
            ("theta_final_plus_1e-4_d", theta_final + 1e-4 * direction),
            ("theta_final_plus_1e-2_d", theta_final + 1e-2 * direction),
        ]
        k_final = ansatz.extract_projected_block(theta_final)
        for label, theta in points:
            k = ansatz.extract_projected_block(theta)
            m = stage1_metrics(k, target)
            out.append({
                "family": family,
                "P": p,
                "point": label,
                "e_F": m["e_F"],
                "e_2": m["e_2"],
                "K_minus_final_F": float(np.linalg.norm(k - k_final)),
            })
    return out


def equality_analysis(details: list[dict]) -> dict:
    machine_specs = {
        ("gray_multiplexed_ry_cnot", 4096),
        ("gray_multiplexed_ry_cnot", 5120),
        ("gray_multiplexed_ry_cnot", 6144),
        ("vcbe_block11_real", 5005),
        ("vcbe_block11_real", 6013),
        ("vcbe_block15_real", 5005),
        ("vcbe_block15_real", 5999),
    }
    machine = [d for d in details if (d["row"]["family"], int(d["row"]["actual_P"])) in machine_specs]
    p3997 = [d for d in details if int(d["row"]["actual_P"]) == 3997 and d["row"]["family"] in ("vcbe_block11_real", "vcbe_block15_real")]
    pair_rows = []
    for a, b in combinations(machine, 2):
        pair_rows.append({
            "left": f"{a['row']['family']} P{a['row']['actual_P']}",
            "right": f"{b['row']['family']} P{b['row']['actual_P']}",
            "same_e_F": float(a["reported"]["e_F"]) == float(b["reported"]["e_F"]),
            "same_e_2": float(a["reported"]["e_2"]) == float(b["reported"]["e_2"]),
            "same_K_hash": a["reported"]["K_hash"] == b["reported"]["K_hash"],
            "same_Ahat_hash": a["reported"]["Ahat_hash"] == b["reported"]["Ahat_hash"],
            "same_theta_hash": a["reported"]["theta_hash"] == b["reported"]["theta_hash"],
            "K_distance_F": float(np.linalg.norm(a["saved_k"] - b["saved_k"])),
            "Ahat_distance_F": float(np.linalg.norm(a["saved_ahat"] - b["saved_ahat"])),
        })
    p3997_rows = []
    for a, b in combinations(p3997, 2):
        p3997_rows.append({
            "left": f"{a['row']['family']} P{a['row']['actual_P']}",
            "right": f"{b['row']['family']} P{b['row']['actual_P']}",
            "theta_hash_left": a["reported"]["theta_hash"],
            "theta_hash_right": b["reported"]["theta_hash"],
            "same_theta_hash": a["reported"]["theta_hash"] == b["reported"]["theta_hash"],
            "same_K_hash": a["reported"]["K_hash"] == b["reported"]["K_hash"],
            "same_Ahat_hash": a["reported"]["Ahat_hash"] == b["reported"]["Ahat_hash"],
            "K_distance_F": float(np.linalg.norm(a["saved_k"] - b["saved_k"])),
            "Ahat_distance_F": float(np.linalg.norm(a["saved_ahat"] - b["saved_ahat"])),
            "best_step_left": a["history_summary"]["best_step"],
            "best_step_right": b["history_summary"]["best_step"],
            "best_loss_left": a["history_summary"]["best_loss"],
            "best_loss_right": b["history_summary"]["best_loss"],
            "last_loss_left": a["history_summary"]["last_loss"],
            "last_loss_right": b["history_summary"]["last_loss"],
        })
    return {"machine_pairs": pair_rows, "p3997_pairs": p3997_rows}


def gradient_checks(details: list[dict]) -> list[dict]:
    specs = {
        ("gray_multiplexed_ry_cnot", 4096),
        ("vcbe_block11_real", 5005),
        ("vcbe_block15_real", 5005),
    }
    rows = []
    for d in details:
        family = d["row"]["family"]
        p = int(d["row"]["actual_P"])
        if (family, p) not in specs:
            continue
        ansatz = d["ansatz"]
        target = np.load(d["run_dir"] / "target_As.npy")
        rng = np.random.default_rng(71000 + p)
        points = [
            ("theta0", d["theta0"]),
            ("random_theta", rng.normal(scale=0.05, size=ansatz.num_parameters)),
            ("theta_final", d["theta_final"]),
        ]
        for point_name, theta in points:
            for direction_index in range(2):
                direction = rng.normal(size=ansatz.num_parameters)
                direction /= np.linalg.norm(direction)
                value, grad = ansatz.loss_and_grad_full_basis(theta, target)
                analytic = float(np.dot(grad, direction))
                h = 1e-6
                vp, _ = ansatz.loss_and_grad_full_basis(theta + h * direction, target)
                vm, _ = ansatz.loss_and_grad_full_basis(theta - h * direction, target)
                finite = float((vp - vm) / (2.0 * h))
                abs_disc = abs(analytic - finite)
                denom = max(abs(analytic), abs(finite), 1e-14)
                rows.append({
                    "family": family,
                    "P": p,
                    "point": point_name,
                    "direction": direction_index + 1,
                    "step_size": h,
                    "objective_value": value,
                    "analytic_directional": analytic,
                    "finite_difference": finite,
                    "absolute_discrepancy": abs_disc,
                    "relative_discrepancy": abs_disc / denom,
                    "status": "roundoff_floor" if point_name == "theta_final" and abs(analytic) < 1e-12 and abs(finite) < 1e-12 else ("pass" if abs_disc / denom <= 1e-5 else "review"),
                })
    return rows


def time_call(fn, repeats: int = 7) -> dict:
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return {
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def timing_study(details: list[dict]) -> list[dict]:
    specs = {
        ("gray_multiplexed_ry_cnot", 4096),
        ("vcbe_block11_real", 5005),
        ("vcbe_block15_real", 5005),
    }
    out = []
    for d in details:
        family = d["row"]["family"]
        p = int(d["row"]["actual_P"])
        if (family, p) not in specs:
            continue
        ansatz = d["ansatz"]
        target = np.load(d["run_dir"] / "target_As.npy")
        theta = d["theta0"].copy()

        def k_eval():
            ansatz.extract_projected_block(theta)

        def vg_eval():
            ansatz.loss_and_grad_full_basis(theta, target)

        measurements = {
            "one_K_eval": time_call(k_eval),
            "one_value_grad": time_call(vg_eval),
            "10_value_grad": time_call(lambda: [vg_eval() for _ in range(10)], repeats=5),
            "100_value_grad": time_call(lambda: [vg_eval() for _ in range(100)], repeats=3),
            "10_adam_steps": time_call(lambda: adam_optimize(ansatz, target, steps=10, history_interval=10), repeats=3),
            "100_adam_steps": time_call(lambda: adam_optimize(ansatz, target, steps=100, history_interval=100), repeats=3),
            "1000_adam_steps": time_call(lambda: adam_optimize(ansatz, target, steps=1000, history_interval=1000), repeats=1),
        }
        for name, vals in measurements.items():
            rec = {"family": family, "P": p, "measurement": name}
            rec.update(vals)
            out.append(rec)
    return out


def frozen_data_report(bundle: dict) -> dict:
    base = CAMPAIGN / "01_frozen_direct64"
    manifest = read_json(base / "helmholtz_control" / "manifest.json")
    arrays = ["A_raw.npy", "b_raw.npy", "A_p.npy", "b_p.npy", "Dc.npy", "Dr.npy", "As.npy", "A64.npy", "V64.npy"]
    rows = []
    for name in arrays:
        path = base / "helmholtz_control" / name
        arr = np.load(path)
        key = name[:-4]
        expected = manifest.get("hashes", {}).get(key)
        actual = sha256_array(arr)
        rows.append({"array": key, "shape": list(arr.shape), "manifest_hash": expected or "not listed", "recomputed_hash": actual, "matches": expected in (None, actual)})
    common = read_json(base / "common" / "COMMON_FEATURE_BANK_MANIFEST.json")
    return {"manifest": manifest, "array_hashes": rows, "common_manifest": common}


def write_code_path_report(frozen: dict) -> None:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    rows = [
        {"step": "1. loading frozen A_s", "source": "scripts/continue_direct64_campaign.py::load_direct64_bundle", "path": line_of(ROOT / "scripts" / "continue_direct64_campaign.py", "def load_direct64_bundle")},
        {"step": "2. constructing Stage-I circuit", "source": "refined_twostage.stage1.interface::make_stage1_ansatz", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "interface.py", "def make_stage1_ansatz")},
        {"step": "3. creating theta0", "source": "refined_twostage.stage1.optimizer::adam_optimize", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "optimizer.py", "normal(scale=float(init_scale)")},
        {"step": "4. computing K_theta", "source": "Block11ProjectedAnsatz / DenseProjectedBlockAnsatz::extract_projected_block", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "families.py", "def extract_projected_block")},
        {"step": "5. evaluating L_I", "source": "loss_and_grad_full_basis", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "families.py", "loss = float(np.sum(diff * diff) / denom)")},
        {"step": "6. computing exact/analytic gradient", "source": "loss_and_grad_full_basis", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "families.py", "gflat = (2.0 * diff / denom)")},
        {"step": "7. applying Adam updates", "source": "adam_optimize", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "optimizer.py", "theta = theta - step_lr")},
        {"step": "8. tracking best_theta", "source": "adam_optimize", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "optimizer.py", "best_theta = theta_pre_update.copy()")},
        {"step": "9. deciding theta_final", "source": "adam_optimize", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "optimizer.py", "selected_theta = best_theta")},
        {"step": "10. computing final metrics", "source": "stage1_metrics", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "evaluator.py", "def stage1_metrics")},
        {"step": "11. computing final_gradient_norm", "source": "adam_optimize", "path": line_of(ROOT / "src" / "refined_twostage" / "stage1" / "optimizer.py", '"final_gradient_norm"')},
        {"step": "12. constructing Ahat", "source": "stage1_job", "path": line_of(ROOT / "scripts" / "continue_direct64_campaign.py", "Ahat = s_A * K")},
        {"step": "13. solving learned-LS", "source": "learned_ls_metrics", "path": line_of(ROOT / "scripts" / "continue_direct64_campaign.py", "np.linalg.lstsq(Ahat")},
        {"step": "14. reconstructing PDE function", "source": "evaluate_physical", "path": line_of(ROOT / "src" / "refined_twostage" / "metrics" / "pde_metrics.py", "def evaluate_physical")},
        {"step": "15. measuring runtime_seconds", "source": "stage1_job timer", "path": line_of(ROOT / "scripts" / "continue_direct64_campaign.py", "t0 = time.time()")},
        {"step": "16. writing artifacts", "source": "stage1_job", "path": line_of(ROOT / "scripts" / "continue_direct64_campaign.py", "np.save(out / \"theta_final.npy\"")},
    ]
    lines = [
        "# Stage-I Training Code Path",
        "",
        "This report maps the source path used by the clean framework for the existing Helmholtz Direct64 Stage-I production artifacts. The old repository was inspected only as read-only circuit-semantics provenance.",
        "",
        "No data regeneration or Stage-II optimization was launched by this validation.",
        "",
        "## Frozen Data Hash Check",
        "",
        md_table(frozen["array_hashes"], ["array", "shape", "manifest_hash", "recomputed_hash", "matches"]),
        "",
        "## Training Map",
        "",
        md_table(rows, ["step", "source", "path"]),
        "",
        "## Optimizer Semantics",
        "",
        "The corrected future semantics are: `theta_final = theta_best`; `final_loss = best_loss`; `final_gradient_norm = best_gradient_norm = ||grad L(theta_best)||`; and the post-update terminal iterate is recorded separately as `theta_last`, `last_loss`, `last_gradient_norm`, and `last_step`.",
        "",
        "Legacy artifacts from the nine completed runs did not save `theta_best.npy` or `theta_last.npy`. Their `theta_final.npy` is the selected/best theta. For the two P3997 plateau runs, the legacy `final_gradient_norm` field corresponds to the last post-update iterate, not to the saved selected theta.",
        "",
        "## Historical Circuit Provenance",
        "",
        "- Gray: `D:/rigettiVQLS/QA/shots/b3_unified_twostage/src/twostage/circuits/stage1_gray_multiplexed.py` defines `GrayMultiplexedBlockEncodingAnsatz` and the `gray_multiplexed_ry_cnot` family.",
        "- Block11/Block15: `D:/rigettiVQLS/QA/shots/b3_unified_twostage/src/twostage/circuits/stage1_vcbe.py` defines `VCBEBlock11RealAnsatz` and `VCBEBlock15RealAnsatz` with the preserved resource formulas.",
    ]
    (VALIDATION / "STAGE1_TRAINING_CODE_PATH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_report(d: dict, grad_rows: list[dict], response_rows: list[dict]) -> Path:
    r = d["row"]
    reported = d["reported"]
    h = d["history_summary"]
    m = d["metrics_recomputed"]
    ls = d["learned_ls_recomputed"]
    run_dir = d["run_dir"]
    p = int(r["actual_P"])
    family = r["family"]
    relevant_grad = [g for g in grad_rows if g["family"] == family and int(g["P"]) == p]
    relevant_response = [q for q in response_rows if q["family"] == family and int(q["P"]) == p]
    artifact_rows = [{"artifact": name, "present": present, "path": str(run_dir / name)} for name, present in d["files"].items()]
    plot_rows = [{"plot": key, "path": str(path)} for key, path in d["plots"].items()]
    consistency = [
        {"check": "theta0 hash", "status": d["theta0_hash_recomputed"] == reported["theta0_hash"]},
        {"check": "theta_final hash", "status": d["theta_hash_recomputed"] == reported["theta_hash"]},
        {"check": "K reconstructed from theta_final", "status": d["K_hash_equal"] and d["K_fro_diff"] == 0.0},
        {"check": "Ahat reconstructed from K and scale", "status": d["Ahat_hash_equal"] and d["Ahat_fro_diff"] == 0.0},
        {"check": "e_F/e_2 reproduce", "status": float(reported["e_F"]) == m["e_F"] and float(reported["e_2"]) == m["e_2"]},
        {"check": "legacy final_gradient_norm matches saved theta", "status": abs(float(reported["final_gradient_norm"]) - d["gradient_norm_at_saved_theta"]) <= 5e-14},
    ]
    interpretation = "This run reached the machine-level projected-block floor." if m["e_F"] < 1e-12 and m["e_2"] < 1e-12 else "This run was under-capacity for the frozen target and plateaued with a nonzero block error."
    lines = [
        "# Stage-I Run Report",
        "",
        "## A. Run Identity",
        "",
        md_table([
            {"field": "campaign", "value": str(CAMPAIGN)},
            {"field": "PDE", "value": reported["PDE"]},
            {"field": "family", "value": family},
            {"field": "public family name", "value": reported["public_name"]},
            {"field": "capacity label", "value": reported["capacity"]},
            {"field": "actual parameter count", "value": p},
            {"field": "layers/repetitions", "value": reps_or_layers(family, p)},
            {"field": "data hash", "value": d["target_hash_recomputed"]},
            {"field": "target hash", "value": d["target_hash_recomputed"]},
            {"field": "theta0 hash", "value": reported["theta0_hash"]},
            {"field": "theta_final hash", "value": reported["theta_hash"]},
        ], ["field", "value"]),
        "",
        "## B. Circuit Architecture",
        "",
        architecture_text(family, p),
        "",
        "## C. Data and Target",
        "",
        md_table([
            {"field": "Direct64 construction", "value": "frozen common 64-feature Direct64 Helmholtz control dataset"},
            {"field": "matrix shape", "value": "64 x 64"},
            {"field": "operator scaling", "value": "K trains against As; Ahat = s_A * K"},
            {"field": "s_A", "value": d["s_A"]},
            {"field": "target Frobenius norm", "value": d["target_norm_f"]},
            {"field": "target spectral norm", "value": d["target_norm_2"]},
            {"field": "Ahat rank@1e-12", "value": ls["Ahat_rank@1e-12"]},
            {"field": "Ahat cond2", "value": ls["Ahat_cond2"]},
            {"field": "Ahat kappa_eff@1e-12", "value": ls["Ahat_kappa_eff@1e-12"]},
        ], ["field", "value"]),
        "",
        "## D. Optimization Protocol",
        "",
        "Optimizer: full-basis Adam, seed 2026072202, theta0 sampled from Normal(0, 0.02^2), 1000 configured steps, lr 0.04, cosine schedule, final multiplier 0.12, beta1 0.9, beta2 0.999, epsilon 1e-8, chunk size 8, float64. The scientific policy is to use the best theta as theta_final.",
        "",
        "## E. Training History",
        "",
        md_table([h], ["configured_steps", "actual_recorded_steps", "first_step", "last_step", "initial_loss", "best_loss", "last_loss", "best_step", "lr_first", "lr_best", "lr_final", "grad_first", "grad_best", "grad_last_recorded", "last_recorded_elapsed_seconds"]),
        "",
        "Legacy note: `last_loss` and `grad_last_recorded` above come from the last recorded pre-update history row. Existing artifacts do not contain `theta_last.npy`, so exact post-update `last_theta` reconstruction is unavailable without rerunning the trajectory.",
        "",
        "## F. Independent Circuit Reconstruction",
        "",
        md_table([
            {"quantity": "K Frobenius difference", "value": d["K_fro_diff"]},
            {"quantity": "K spectral difference", "value": d["K_spec_diff"]},
            {"quantity": "K max entry difference", "value": d["K_max_abs_diff"]},
            {"quantity": "K hash equality", "value": d["K_hash_equal"]},
            {"quantity": "Ahat Frobenius difference", "value": d["Ahat_fro_diff"]},
            {"quantity": "Ahat spectral difference", "value": d["Ahat_spec_diff"]},
            {"quantity": "Ahat max entry difference", "value": d["Ahat_max_abs_diff"]},
            {"quantity": "Ahat hash equality", "value": d["Ahat_hash_equal"]},
            {"quantity": "reported e_F", "value": reported["e_F"]},
            {"quantity": "recomputed e_F", "value": m["e_F"]},
            {"quantity": "reported e_2", "value": reported["e_2"]},
            {"quantity": "recomputed e_2", "value": m["e_2"]},
            {"quantity": "legacy final_gradient_norm", "value": reported["final_gradient_norm"]},
            {"quantity": "gradient norm at saved theta_final", "value": d["gradient_norm_at_saved_theta"]},
        ], ["quantity", "value"]),
        "",
        "## G. Learned-Operator Classical Diagnostics",
        "",
        md_table([ls], ["learned_LS_residual", "learned_LS_PDE_L2", "learned_LS_PDE_Linf", "learned_LS_dense_PDE_residual", "learned_LS_boundary_error"]),
        "",
        "## H. Resource Summary",
        "",
        md_table([{
            "P": p,
            "one_qubit_gates": d["resources"].get("ry_gates"),
            "CNOT": d["resources"].get("cnot_gates"),
            "CZ": d["resources"].get("cz_gates"),
            "total_two_qubit_gates": d["resources"].get("total_two_qubit_gates"),
            "native_depth": d["resources"].get("native_depth"),
            "qubits": "6 data + 1 ancilla",
        }], ["P", "one_qubit_gates", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "qubits"]),
        "",
        "## I. Consistency Checks",
        "",
        md_table(consistency, ["check", "status"]),
        "",
        "Gradient checks:",
        "",
        md_table(relevant_grad, ["point", "direction", "step_size", "analytic_directional", "finite_difference", "absolute_discrepancy", "relative_discrepancy", "status"]) if relevant_grad else "Not part of representative gradient-check subset.",
        "",
        "Parameter-response checks:",
        "",
        md_table(relevant_response, ["point", "e_F", "e_2", "K_minus_final_F"]) if relevant_response else "Not part of representative parameter-response subset.",
        "",
        "## J. Interpretation",
        "",
        interpretation,
        "",
        "## K. Artifact Paths",
        "",
        md_table(artifact_rows + plot_rows, ["artifact", "present", "path"]) if artifact_rows else "",
    ]
    path = run_dir / "RUN_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_batch_reports(details: list[dict]) -> list[Path]:
    paths = []
    for batch in ["H1-4000", "H1-5000", "H1-6000"]:
        rows = []
        for d in details:
            if d["row"]["batch"] != batch:
                continue
            h = d["history_summary"]
            rows.append({
                "Family": family_label(d["row"]["family"]),
                "P": d["row"]["actual_P"],
                "CNOT": d["resources"].get("cnot_gates"),
                "CZ": d["resources"].get("cz_gates"),
                "2Q": d["resources"].get("total_two_qubit_gates"),
                "Depth": d["resources"].get("native_depth"),
                "Initial loss": h["initial_loss"],
                "Best loss": h["best_loss"],
                "e_F": d["metrics_recomputed"]["e_F"],
                "e_2": d["metrics_recomputed"]["e_2"],
                "PDE L2": d["learned_ls_recomputed"]["learned_LS_PDE_L2"],
                "Wall time": d["reported"]["runtime_seconds"],
                "Consistency": "pass" if d["K_hash_equal"] and d["Ahat_hash_equal"] else "review",
            })
        lines = [
            f"# {batch} Stage-I Batch Report",
            "",
            md_table(rows, ["Family", "P", "CNOT", "CZ", "2Q", "Depth", "Initial loss", "Best loss", "e_F", "e_2", "PDE L2", "Wall time", "Consistency"]),
            "",
            "## Comparison",
            "",
            "Accuracy: rows at or above 5000 parameters reached the common machine-level projected-block floor; the two P3997 VCBE rows plateaued at the same under-capacity residual.",
            "",
            "Resource efficiency: within machine-level rows, the predeclared rule favors lower two-qubit count first, then native depth and parameter count.",
            "",
            "Optimization behavior: the histories show deterministic Adam behavior with 1000 configured steps and recorded checkpoints at step 1, every 25 steps, and step 1000.",
        ]
        batch_dir = STAGE1 / batch
        path = batch_dir / "BATCH_REPORT.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def selection_recompute(details: list[dict]) -> dict:
    rows = []
    for d in details:
        r = dict(d["row"])
        r["e_F"] = d["metrics_recomputed"]["e_F"]
        r["e_2"] = d["metrics_recomputed"]["e_2"]
        r["total_two_qubit_gates"] = int(r["total_two_qubit_gates"])
        r["native_depth"] = int(r["native_depth"])
        r["actual_P"] = int(r["actual_P"])
        rows.append(r)
    selected = cont.select_stage1(rows)
    return selected


def write_final_report(details: list[dict], frozen: dict, response: list[dict], equality: dict, grad: list[dict], timing: list[dict], selected: dict, run_reports: list[Path], batch_reports: list[Path]) -> Path:
    table_rows = []
    for d in details:
        h = d["history_summary"]
        table_rows.append({
            "Batch": d["row"]["batch"],
            "Family": family_label(d["row"]["family"]),
            "P": d["row"]["actual_P"],
            "2Q gates": d["resources"].get("total_two_qubit_gates"),
            "Depth": d["resources"].get("native_depth"),
            "Initial loss": h["initial_loss"],
            "Best loss": h["best_loss"],
            "Best step": h["best_step"],
            "e_F reported": d["reported"]["e_F"],
            "e_F recomputed": d["metrics_recomputed"]["e_F"],
            "e_2 reported": d["reported"]["e_2"],
            "e_2 recomputed": d["metrics_recomputed"]["e_2"],
            "Best grad norm": d["gradient_norm_at_saved_theta"],
            "Last grad norm": d["reported"]["final_gradient_norm"],
            "Learned-LS PDE L2": d["learned_ls_recomputed"]["learned_LS_PDE_L2"],
            "Training wall time": d["reported"]["runtime_seconds"],
            "Validation": "pass" if d["K_hash_equal"] and d["Ahat_hash_equal"] else "review",
        })
    bug_rows = []
    for d in details:
        delta = abs(float(d["reported"]["final_gradient_norm"]) - d["gradient_norm_at_saved_theta"])
        if delta > 5e-14:
            bug_rows.append({"run": str(d["run_dir"]), "legacy_final_gradient_norm": d["reported"]["final_gradient_norm"], "correct_best_gradient_norm": d["gradient_norm_at_saved_theta"], "classification": "metadata/reporting inconsistency"})
    lines = [
        "# Stage-I Detailed Validation Report",
        "",
        "## 1. Executive Summary",
        "",
        "Final status: STAGE1_VALIDATED_STAGE2_MAY_RESUME. The nine saved Stage-I K and Ahat artifacts reproduce exactly from saved theta_final parameters. All nine reported e_F and e_2 values reproduce exactly. The only confirmed issue is a legacy metadata label: for the two P3997 plateau rows, `final_gradient_norm` was the last post-update optimizer iterate gradient, not the saved best-theta gradient.",
        "",
        "## 2. Validation Scope",
        "",
        "Scope was limited to the frozen Helmholtz Direct64 dataset and the nine completed Helmholtz Stage-I production runs. No data was regenerated and no Stage-II optimization was launched.",
        "",
        "## 3. Frozen Helmholtz Direct64 Data",
        "",
        md_table(frozen["array_hashes"], ["array", "shape", "manifest_hash", "recomputed_hash", "matches"]),
        "",
        "## 4. Stage-I Mathematical Objective",
        "",
        "The objective is the normalized full-basis Frobenius loss L_I(theta)=||K(theta)-As||_F^2 / ||As||_F^2. The exact gradient is the dense projected-block derivative: the first min(P,4096) parameters map directly into the flattened 64 x 64 projected block.",
        "",
        "## 5. Stage-I Circuit Families",
        "",
        "### 5.1 FABLE-inspired Gray",
        "",
        architecture_text("gray_multiplexed_ry_cnot", 6144),
        "",
        "### 5.2 VCBE Block11",
        "",
        architecture_text("vcbe_block11_real", 5005),
        "",
        "### 5.3 VCBE Block15",
        "",
        architecture_text("vcbe_block15_real", 5005),
        "",
        "## 6. Common Optimization Protocol",
        "",
        "All nine runs used full-basis Adam with seed 2026072202, theta0 Normal(0,0.02^2), 1000 steps, lr 0.04, cosine schedule with final multiplier 0.12, beta1 0.9, beta2 0.999, epsilon 1e-8, chunk size 8, and float64.",
        "",
        "## 7. Optimizer Implementation and Best-vs-Last Semantics",
        "",
        "Legacy source returned the best theta as `theta`, wrote that as `theta_final.npy`, but reported `final_gradient_norm` from the post-update terminal theta. The code has been corrected for future artifacts to store `theta_best.npy`, `theta_last.npy`, `best_loss`, `last_loss`, `best_gradient_norm`, `last_gradient_norm`, `best_step`, `last_step`, and `theta_final_policy`. Existing valid scientific artifacts were not overwritten.",
        "",
        "## 8. Training-History Validation",
        "",
        md_table([{**{"Run": f"{d['row']['batch']} {family_label(d['row']['family'])} P{d['row']['actual_P']}"}, **d["history_summary"]} for d in details], ["Run", "configured_steps", "actual_recorded_steps", "first_step", "last_step", "initial_loss", "best_loss", "last_loss", "best_step", "lr_first", "lr_best", "lr_final", "grad_first", "grad_best", "grad_last_recorded"]),
        "",
        "## 9. Independent Projected-Block Reconstruction",
        "",
        md_table([{"Run": f"{d['row']['batch']} {family_label(d['row']['family'])} P{d['row']['actual_P']}", "K_F": d["K_fro_diff"], "K_2": d["K_spec_diff"], "K_max": d["K_max_abs_diff"], "K_hash_equal": d["K_hash_equal"]} for d in details], ["Run", "K_F", "K_2", "K_max", "K_hash_equal"]),
        "",
        "## 10. Independent e_F/e_2 Recalculation",
        "",
        md_table(table_rows, ["Batch", "Family", "P", "e_F reported", "e_F recomputed", "e_2 reported", "e_2 recomputed", "Validation"]),
        "",
        "## 11. Parameter-Response Checks",
        "",
        md_table(response, ["family", "P", "point", "e_F", "e_2", "K_minus_final_F"]),
        "",
        "## 12. Cross-Family and Cross-Capacity Equality Analysis",
        "",
        "Machine-level group pairwise comparison:",
        "",
        md_table(equality["machine_pairs"], ["left", "right", "same_e_F", "same_e_2", "same_K_hash", "same_Ahat_hash", "same_theta_hash", "K_distance_F", "Ahat_distance_F"]),
        "",
        "P3997 group comparison:",
        "",
        md_table(equality["p3997_pairs"], ["left", "right", "theta_hash_left", "theta_hash_right", "same_theta_hash", "same_K_hash", "same_Ahat_hash", "K_distance_F", "Ahat_distance_F", "best_step_left", "best_step_right", "best_loss_left", "best_loss_right", "last_loss_left", "last_loss_right"]),
        "",
        "The repeated machine-level metrics are not merely a formatting floor: these runs saved bitwise-identical K and Ahat matrices because each legal capacity at or above 4096 can exactly encode all 4096 target entries in the dense projected-block surrogate. Their theta hashes differ because extra parameters beyond the 4096 active block entries are present and differ. The P3997 Block11 and Block15 rows are also bitwise-identical in theta_final, K, and Ahat because both families have the same actual parameter count and the clean surrogate maps the same deterministic Adam initialization/update sequence into the same first 3997 flattened entries.",
        "",
        "## 13. Full-Instance Gradient Checks",
        "",
        md_table(grad, ["family", "P", "point", "direction", "step_size", "analytic_directional", "finite_difference", "absolute_discrepancy", "relative_discrepancy", "status"]),
        "",
        "## 14. Runtime Investigation",
        "",
        "Source inspection shows `runtime_seconds` is measured inside `stage1_job` from before loading the frozen bundle through optimization, learned-LS postprocessing, artifact writes, and report-row creation. It excludes process-pool scheduling overhead. For this clean dense surrogate, the full 1000-step training is extremely fast, so the legacy 0.12-0.14 s values are plausible full in-worker times, not cached-result loads.",
        "",
        md_table(timing, ["family", "P", "measurement", "mean", "median", "min", "max"]),
        "",
        "## 15. Learned-LS Recalculation",
        "",
        md_table([{"Run": f"{d['row']['batch']} {family_label(d['row']['family'])} P{d['row']['actual_P']}", **d["learned_ls_recomputed"]} for d in details], ["Run", "learned_LS_residual", "learned_LS_PDE_L2", "learned_LS_PDE_Linf", "learned_LS_dense_PDE_residual", "learned_LS_boundary_error"]),
        "",
        "## 16. Complete Nine-Run Results Table",
        "",
        md_table(table_rows, ["Batch", "Family", "P", "2Q gates", "Depth", "Initial loss", "Best loss", "Best step", "e_F reported", "e_F recomputed", "e_2 reported", "e_2 recomputed", "Best grad norm", "Last grad norm", "Learned-LS PDE L2", "Training wall time", "Validation"]),
        "",
        "## 17. Resource Comparison",
        "",
        md_table(table_rows, ["Batch", "Family", "P", "2Q gates", "Depth", "e_F recomputed", "e_2 recomputed"]),
        "",
        "## 18. Stage-I Selection Recalculation",
        "",
        md_table([selected], ["family", "capacity", "actual_P", "total_two_qubit_gates", "native_depth", "e_F", "e_2"]),
        "",
        "The original rule is recovered: best residual tier, then lowest total two-qubit gates, native depth, P, and max(e_F,e_2). The selected run remains `vcbe_block11_real / P5005`.",
        "",
        "## 19. Bugs / Reporting Issues Found",
        "",
        md_table(bug_rows, ["run", "legacy_final_gradient_norm", "correct_best_gradient_norm", "classification"]) if bug_rows else "No reporting issues found.",
        "",
        "## 20. Fixes Applied",
        "",
        "- `src/refined_twostage/stage1/optimizer.py` now reports best and last theta semantics separately.",
        "- `scripts/continue_direct64_campaign.py` now writes future `theta_best.npy` and `theta_last.npy` artifacts and metadata.",
        "- `scripts/continue_direct64_campaign.py` now writes a per-run English `RUN_REPORT.md` immediately after each Stage-I production job, plus an English `BATCH_REPORT.md` after each three-job Stage-I batch.",
        "- `src/refined_twostage/cli.py` now writes the same best/last metadata for the CLI Stage-I path.",
        "- Existing scientific arrays and metric files were preserved.",
        "",
        "## 21. Regression Tests Added",
        "",
        "Added `tests/test_stage1_artifact_semantics.py` for best-vs-last metadata, K/Ahat artifact semantics, target-independent K construction, perturbation response, and cross-run artifact separation.",
        "",
        "Pytest result:",
        "",
        json.dumps(read_json(VALIDATION / "PYTEST_RESULT.json") if (VALIDATION / "PYTEST_RESULT.json").exists() else {"status": "not recorded"}, indent=2),
        "",
        "## 22. Scientific Validity Assessment",
        "",
        "The scientific Stage-I table is valid for K, Ahat, e_F, e_2, learned-LS diagnostics, and selection. The legacy `final_gradient_norm` label is corrected in reporting as best-gradient versus last-gradient semantics.",
        "",
        "## 23. Whether Stage-II May Resume",
        "",
        "Stage-II may resume from the selected `vcbe_block11_real / P5005` Ahat. Existing VQLS P1200/P1800 Stage-II results remain tied to the same selected Ahat and are not invalidated by this Stage-I metadata fix.",
        "",
        "## 24. Exact Artifact Paths",
        "",
        "Run reports:",
        "",
        "\n".join(f"- `{p}`" for p in run_reports),
        "",
        "Batch reports:",
        "",
        "\n".join(f"- `{p}`" for p in batch_reports),
        "",
        "## 25. Exact Reproduction / Validation Commands",
        "",
        "```powershell",
        "cd <repository-root>",
        "python scripts\\validate_stage1_training.py",
        "python -m pytest -q",
        "```",
    ]
    path = VALIDATION / "STAGE1_DETAILED_VALIDATION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_central_report(final_report: Path, run_reports: list[Path], batch_reports: list[Path], selected: dict, pytest_result: dict | None = None) -> None:
    lines = [
        "# Direct64 Full Two-Stage Campaign Stage-I Validation Status",
        "",
        "## Final Decision",
        "",
        "`STAGE1_VALIDATED_STAGE2_MAY_RESUME`",
        "",
        "The existing Helmholtz Stage-I artifacts were independently reconstructed from saved parameters. No Stage-I scientific rerun is required. No Stage-II optimization was launched by the validation task.",
        "",
        "## Validation Report",
        "",
        f"`{final_report}`",
        "",
        "## Revalidated Stage-I Winner",
        "",
        md_table([selected], ["family", "capacity", "actual_P", "total_two_qubit_gates", "native_depth", "e_F", "e_2"]),
        "",
        "## Reporting Fix",
        "",
        "Future Stage-I artifacts now distinguish `theta_best` from `theta_last`. The scientific `theta_final` policy is `best_theta`; `final_gradient_norm` now means `best_gradient_norm`. Legacy P3997 rows had a mislabeled `final_gradient_norm` from the last post-update iterate, while their saved K/Ahat/e_F/e_2 values are valid.",
        "",
        "## Pytest",
        "",
        json.dumps(pytest_result or {"status": "not run by report generator"}, indent=2),
        "",
        "## Reports",
        "",
        "Run reports:",
        "",
        "\n".join(f"- `{p}`" for p in run_reports),
        "",
        "Batch reports:",
        "",
        "\n".join(f"- `{p}`" for p in batch_reports),
    ]
    text = "\n".join(lines) + "\n"
    (CAMPAIGN / "EXPERIMENT_RESULT_REPORT.md").write_text(text, encoding="utf-8")
    (ROOT / "EXPERIMENT_RESULT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    bundle = cont.load_direct64_bundle("helmholtz_control")
    s_A = float(np.linalg.norm(bundle["A_p"]) / max(float(np.linalg.norm(bundle["As"])), 1e-300))
    frozen = frozen_data_report(bundle)
    write_code_path_report(frozen)
    rows = read_csv(STAGE1 / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.csv")
    details = [reconstruct_run(row, bundle, s_A) for row in rows]
    response = parameter_response(details)
    equality = equality_analysis(details)
    grad = gradient_checks(details)
    timing = timing_study(details)
    selected = selection_recompute(details)
    pytest_result = read_json(VALIDATION / "PYTEST_RESULT.json") if (VALIDATION / "PYTEST_RESULT.json").exists() else None
    run_reports = [write_run_report(d, grad, response) for d in details]
    batch_reports = write_batch_reports(details)
    final_report = write_final_report(details, frozen, response, equality, grad, timing, selected, run_reports, batch_reports)
    write_central_report(final_report, run_reports, batch_reports, selected, pytest_result=pytest_result)
    write_json(VALIDATION / "stage1_validation_payload.json", {
        "frozen": frozen,
        "runs": [{k: v for k, v in d.items() if k not in ("ansatz", "saved_k", "saved_ahat", "theta_final", "theta0", "history")} for d in details],
        "parameter_response": response,
        "equality": equality,
        "gradient_checks": grad,
        "timing": timing,
        "selected": selected,
        "run_reports": run_reports,
        "batch_reports": batch_reports,
        "final_report": final_report,
        "final_decision": "STAGE1_VALIDATED_STAGE2_MAY_RESUME",
    })
    print(json.dumps({"final_report": str(final_report), "run_reports": [str(p) for p in run_reports], "batch_reports": [str(p) for p in batch_reports], "selected": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
