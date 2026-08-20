"""Continuation runner for the frozen Direct64 campaign."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refined_twostage.io.hashing import sha256_array, short_hash_array
from refined_twostage.metrics.matrix_metrics import singular_diagnostics
from refined_twostage.metrics.pde_metrics import evaluate_physical
from refined_twostage.problems.common import get_pde
from refined_twostage.rfm.features import SineFeatureBank
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.objectives import objective_value_and_grad_theta, scalar_eliminated_components
from refined_twostage.stage2.polynomial import nearest_legal_capacity

CAMPAIGN = ROOT / "results" / "direct64_full_twostage_campaign_20260820_204937"


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def md_table(rows: list[dict], fields: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "{:.6e}".format(v) if math.isfinite(v) else str(v)
        return str(v)

    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def load_direct64_bundle(pde_key: str):
    base = CAMPAIGN / "01_frozen_direct64" / pde_key
    common = CAMPAIGN / "01_frozen_direct64" / "common"
    w = np.load(common / "feature_w.npy")
    b = np.load(common / "feature_b.npy")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    alpha = float(manifest["feature_alpha"])
    fb = SineFeatureBank(w=w, b=b, alpha=alpha, seed=int(manifest["feature_seed"]))
    return {
        "pde": get_pde(pde_key),
        "feature_bank": fb,
        "dense_x": np.load(common / "dense_grid.npy"),
        "A_p": np.load(base / "A_p.npy"),
        "b_p": np.load(base / "b_p.npy"),
        "Dc": np.load(base / "Dc.npy"),
        "As": np.load(base / "As.npy"),
        "manifest": manifest,
    }


def raw_coefficients(Dc, z):
    return np.asarray(Dc, dtype=float) * np.asarray(z, dtype=float)


def learned_ls_metrics(bundle, Ahat):
    z, *_ = np.linalg.lstsq(Ahat, bundle["b_p"], rcond=None)
    coeff = raw_coefficients(bundle["Dc"], z)
    phys = evaluate_physical(bundle["pde"], bundle["feature_bank"], coeff, bundle["dense_x"])
    mat = singular_diagnostics(Ahat)
    return {
        "learned_LS_residual": float(np.linalg.norm(Ahat @ z - bundle["b_p"]) / max(float(np.linalg.norm(bundle["b_p"])), 1e-300)),
        "learned_LS_PDE_L2": phys["PDE_relative_L2"],
        "learned_LS_PDE_Linf": phys["PDE_relative_Linf"],
        "learned_LS_dense_PDE_residual": phys["dense_PDE_residual"],
        "learned_LS_boundary_error": phys["boundary_error"],
        "Ahat_rank@1e-12": mat["ranks"]["rank@1e-12"],
        "Ahat_cond2": mat["cond2"],
        "Ahat_kappa_eff@1e-12": mat["effective_condition_numbers"]["kappa_eff@1e-12"],
        "Ahat_singular_values": mat["singular_values"],
    }


def write_stage1_run_report(out: Path, row: dict, config: dict, resources: dict) -> None:
    artifact_names = [
        "theta0.npy",
        "theta_final.npy",
        "theta_best.npy",
        "theta_last.npy",
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
    artifacts = [{"artifact": name, "present": (out / name).exists(), "path": str(out / name)} for name in artifact_names]
    lines = [
        "# Stage-I Run Report",
        "",
        "## Run Identity",
        "",
        md_table([row], ["PDE", "public_name", "family", "batch", "capacity", "actual_P", "theta0_hash", "theta_hash", "K_hash", "Ahat_hash"]),
        "",
        "## Data and Target",
        "",
        "This run used the frozen Helmholtz Direct64 target `As` loaded by `load_direct64_bundle('helmholtz_control')`. The learned physical operator is formed as `Ahat = s_A * K_theta`, with `s_A` recovered from the frozen `A_p` and `As` norms.",
        "",
        "## Optimizer",
        "",
        md_table([config], ["optimizer", "theta0_seed", "theta0_distribution", "steps", "lr", "schedule", "final_multiplier", "chunk_size", "float"]),
        "",
        "The future artifact schema records `theta_best.npy` and `theta_last.npy` separately. The scientific final-theta policy is `theta_final = theta_best`, so `final_gradient_norm` means `best_gradient_norm`; `last_gradient_norm` is retained separately.",
        "",
        "## Metrics",
        "",
        md_table([row], ["e_F", "e_2", "absolute_Frobenius_error", "absolute_spectral_error", "final_loss", "best_loss", "last_loss", "best_step", "last_step", "final_gradient_norm", "best_gradient_norm", "last_gradient_norm", "learned_LS_residual", "learned_LS_PDE_L2", "learned_LS_PDE_Linf", "learned_LS_dense_PDE_residual", "learned_LS_boundary_error", "runtime_seconds"]),
        "",
        "## Resources",
        "",
        md_table([resources], ["parameters", "ry_gates", "cnot_gates", "cz_gates", "total_two_qubit_gates", "total_gates", "native_depth", "logical_qubits", "ancilla_qubits"]),
        "",
        "## Artifacts",
        "",
        md_table(artifacts, ["artifact", "present", "path"]),
    ]
    (out / "RUN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage1_batch_report(batch_dir: Path, batch_name: str, rows: list[dict]) -> None:
    fields = ["public_name", "capacity", "actual_P", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "best_loss", "e_F", "e_2", "learned_LS_PDE_L2", "runtime_seconds"]
    lines = [
        f"# {batch_name} Stage-I Batch Report",
        "",
        md_table(rows, fields),
        "",
        "Accuracy, resource efficiency, and optimizer behavior should be interpreted using the predeclared Stage-I selection rule: best residual tier, then lowest total two-qubit gates, native depth, P, and max(e_F,e_2).",
    ]
    (batch_dir / "BATCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage1_job(job: dict):
    t0 = time.time()
    pde_key = "helmholtz_control"
    bundle = load_direct64_bundle(pde_key)
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "stdout.log").write_text("started\n", encoding="utf-8")
    (out / "stderr.log").write_text("", encoding="utf-8")
    ansatz = make_stage1_ansatz(job["family"], target_parameters=int(job["target_P"]))
    np.save(out / "target_As.npy", bundle["As"])
    config = {
        "pde_key": pde_key,
        "family": job["family"],
        "public_name": job["public_name"],
        "target_P": int(job["target_P"]),
        "actual_P": int(ansatz.num_parameters),
        "optimizer": "full_basis_adam_refined_v1",
        "theta0_seed": 2026072202,
        "theta0_distribution": "Normal(0,0.02^2)",
        "steps": 1000,
        "lr": 0.04,
        "schedule": "cosine",
        "final_multiplier": 0.12,
        "chunk_size": 8,
        "float": "float64",
    }
    (out / "config_snapshot.yaml").write_text("\n".join("{}: {}".format(k, v) for k, v in config.items()) + "\n", encoding="utf-8")
    write_json(out / "status.json", {"status": "running", "started_unix": t0})
    result = adam_optimize(ansatz, bundle["As"], steps=1000, lr=0.04, final_multiplier=0.12, chunk_size=8, history_interval=25)
    K = result["K"]
    s_A = float(bundle["manifest"]["physical_metrics"].get("s_A", 1.0)) if "physical_metrics" in bundle["manifest"] else 1.0
    # Frozen As was saved as A_p/(1.05*||A_p||_2); recover the same scale from A_p and As.
    denom = max(float(np.linalg.norm(bundle["As"])), 1e-300)
    s_A = float(np.linalg.norm(bundle["A_p"]) / denom)
    Ahat = s_A * K
    resources = ansatz.resource_counts()
    ls = learned_ls_metrics(bundle, Ahat)
    row = {
        "PDE": bundle["pde"].name,
        "family": job["family"],
        "public_name": job["public_name"],
        "batch": job["batch"],
        "capacity": job["capacity"],
        "target_P": int(job["target_P"]),
        "actual_P": int(ansatz.num_parameters),
        "CNOT": int(resources.cnot_gates),
        "CZ": int(resources.cz_gates),
        "total_two_qubit_gates": int(resources.total_two_qubit_gates),
        "native_depth": int(resources.native_depth),
        "e_F": result["metrics"]["e_F"],
        "e_2": result["metrics"]["e_2"],
        "absolute_Frobenius_error": result["metrics"]["absolute_frobenius_error"],
        "absolute_spectral_error": result["metrics"]["absolute_spectral_error"],
        "final_loss": result["final_loss"],
        "best_loss": result["best_loss"],
        "last_loss": result["last_loss"],
        "best_step": result["best_step"],
        "last_step": result["last_step"],
        "final_gradient_norm": result["final_gradient_norm"],
        "best_gradient_norm": result["best_gradient_norm"],
        "last_gradient_norm": result["last_gradient_norm"],
        "theta_final_policy": result["theta_final_policy"],
        "theta0_hash": short_hash_array(result["theta0"]),
        "theta_hash": short_hash_array(result["theta"]),
        "theta_best_hash": short_hash_array(result["theta_best"]),
        "theta_last_hash": short_hash_array(result["theta_last"]),
        "K_hash": sha256_array(K),
        "Ahat_hash": sha256_array(Ahat),
        "runtime_seconds": float(time.time() - t0),
        "result_directory": str(out),
    }
    row.update({k: v for k, v in ls.items() if k != "Ahat_singular_values"})
    np.save(out / "theta0.npy", result["theta0"])
    np.save(out / "theta_final.npy", result["theta"])
    np.save(out / "theta_best.npy", result["theta_best"])
    np.save(out / "theta_last.npy", result["theta_last"])
    np.save(out / "K_theta.npy", K)
    np.save(out / "Ahat.npy", Ahat)
    np.save(out / "Ahat_singular_values.npy", np.asarray(ls["Ahat_singular_values"], dtype=float))
    write_csv(out / "training_history.csv", result["history"])
    write_json(out / "training_history.json", {"rows": result["history"]})
    write_json(out / "metrics.json", row)
    resource_payload = resources.to_dict()
    write_json(out / "resource_report.json", resource_payload)
    write_json(out / "hash_manifest.json", {"theta0": row["theta0_hash"], "theta": row["theta_hash"], "theta_best": row["theta_best_hash"], "theta_last": row["theta_last_hash"], "K_theta": row["K_hash"], "Ahat": row["Ahat_hash"], "target_As": sha256_array(bundle["As"])})
    write_json(out / "status.json", {"status": "complete", "completed_unix": time.time(), "metrics": row})
    write_stage1_run_report(out, row, config, resource_payload)
    (out / "stdout.log").write_text("complete\n", encoding="utf-8")
    return row


def select_stage1(rows: list[dict]):
    def tier(row):
        if row["e_F"] <= 1e-10 and row["e_2"] <= 1e-10:
            return 0
        if row["e_F"] <= 1e-8 and row["e_2"] <= 1e-8:
            return 1
        return 2

    return min(rows, key=lambda r: (tier(r), int(r["total_two_qubit_gates"]), int(r["native_depth"]), int(r["actual_P"]), max(float(r["e_F"]), float(r["e_2"]))))


def resource_row(ansatz):
    r = ansatz.resource_counts()
    return {
        "ry_gates": int(getattr(r, "ry_count", getattr(r, "ry_gates", 0))),
        "CNOT": int(getattr(r, "cnot_count", getattr(r, "cnot_gates", 0))),
        "CZ": int(getattr(r, "cz_count", 0)),
        "total_two_qubit_gates": int(getattr(r, "two_qubit_gate_count", getattr(r, "total_two_qubit_gates", 0))),
        "native_depth": int(getattr(r, "native_depth", getattr(r, "greedy_native_critical_path_depth", 0))),
    }


def stage2_diag(bundle, Ahat, ansatz, theta, learned_coeff):
    y = ansatz.state(theta)
    comp = scalar_eliminated_components(Ahat, bundle["b_p"], y)
    z = np.asarray(comp["z"], dtype=float)
    coeff = raw_coefficients(bundle["Dc"], z)
    phys = evaluate_physical(bundle["pde"], bundle["feature_bank"], coeff, bundle["dense_x"])
    learned_values = bundle["feature_bank"].values(bundle["dense_x"]) @ learned_coeff
    stage2_values = bundle["feature_bank"].values(bundle["dense_x"]) @ coeff
    explicit = float(np.linalg.norm(comp["r_vec"]) / max(float(np.linalg.norm(bundle["b_p"])), 1e-300))
    return {
        "objective_value_check": explicit,
        "alpha_star": float(comp["alpha"]),
        "exact_matrix_residual": float(np.linalg.norm(bundle["A_p"] @ z - bundle["b_p"]) / max(float(np.linalg.norm(bundle["b_p"])), 1e-300)),
        "learned_matrix_residual": float(np.linalg.norm(Ahat @ z - bundle["b_p"]) / max(float(np.linalg.norm(bundle["b_p"])), 1e-300)),
        "PDE_L2": phys["PDE_relative_L2"],
        "PDE_Linf": phys["PDE_relative_Linf"],
        "dense_PDE_residual": phys["dense_PDE_residual"],
        "boundary_error": phys["boundary_error"],
        "u_StageII_vs_u_Ahat_LS_relative_L2": float(np.linalg.norm(stage2_values - learned_values) / max(float(np.linalg.norm(learned_values)), 1e-300)),
    }


def run_stage2_attempt(bundle, Ahat, learned_coeff, family: str, target_p: int, out: Path):
    result_path = out / "RESULT.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if family == "alternating_ry_cnot":
        row = {"family": family, "target_P": int(target_p), "layers_or_repetitions": int(target_p) // 6 - 1, "actual_P": int(target_p)}
    else:
        row = nearest_legal_capacity(family, int(target_p))
    ansatz = make_stage2_ansatz(family, layers=int(row["layers_or_repetitions"]))
    out.mkdir(parents=True, exist_ok=True)
    theta0 = np.random.default_rng(13379).normal(scale=0.70, size=int(ansatz.num_parameters)).astype(float)
    np.save(out / "theta0.npy", theta0)
    history = []
    best = {"value": float("inf"), "theta": theta0.copy()}
    evals = 0
    t0 = time.time()
    initial_value, initial_grad, _ = objective_value_and_grad_theta(Ahat, bundle["b_p"], ansatz, theta0, objective="direct_r")

    def fun(theta):
        nonlocal evals, best
        evals += 1
        value, grad, _comp = objective_value_and_grad_theta(Ahat, bundle["b_p"], ansatz, theta, objective="direct_r")
        if value < best["value"]:
            best = {"value": float(value), "theta": np.asarray(theta, dtype=float).copy()}
        if evals == 1 or evals % 25 == 0:
            rec = {"evaluation": int(evals), "direct_r": float(value), "gradient_norm": float(np.linalg.norm(grad)), "best_direct_r": float(best["value"]), "elapsed_seconds": float(time.time() - t0)}
            history.append(rec)
            write_csv(out / "trajectory.csv", history)
            write_json(out / "progress_status.json", rec)
        return float(value), np.asarray(grad, dtype=float)

    write_json(out / "config.json", {"family": family, "target_P": int(target_p), "resolved": row, "objective_name": "direct_r", "theta0_seed": 13379, "init_scale": 0.70, "optimizer": "L-BFGS-B", "maxiter": 30000, "maxfun": 60000, "maxcor": 100, "maxls": 150, "ftol": 1e-16, "gtol": 1e-12})
    result = minimize(fun, theta0, method="L-BFGS-B", jac=True, options={"maxiter": 30000, "maxfun": 60000, "maxcor": 100, "maxls": 150, "ftol": 1e-16, "gtol": 1e-12, "disp": False})
    theta = np.asarray(result.x, dtype=float)
    final_value, final_grad, _ = objective_value_and_grad_theta(Ahat, bundle["b_p"], ansatz, theta, objective="direct_r")
    diag = stage2_diag(bundle, Ahat, ansatz, theta, learned_coeff)
    payload = {
        "family": family,
        "target_P": int(target_p),
        "actual_P": int(ansatz.num_parameters),
        "layers_or_repetitions": int(row["layers_or_repetitions"]),
        "objective_name": "direct_r",
        "initial_r": float(initial_value),
        "initial_gradient_norm": float(np.linalg.norm(initial_grad)),
        "final_r": float(final_value),
        "best_r": float(min(best["value"], final_value)),
        "gradient_norm": float(np.linalg.norm(final_grad)),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", evals)),
        "njev": int(getattr(result, "njev", evals)),
        "runtime_seconds": float(time.time() - t0),
        "termination_message": str(result.message),
        "optimizer_success": bool(result.success),
        "theta0_hash": short_hash_array(theta0),
        "theta_hash": short_hash_array(theta),
        "result_directory": str(out),
    }
    payload.update(resource_row(ansatz))
    payload.update(diag)
    payload["objective_value_matches_explicit_residual"] = bool(abs(payload["final_r"] - payload["objective_value_check"]) <= 1e-10 * max(1.0, abs(payload["final_r"])))
    np.save(out / "final_theta.npy", theta)
    np.save(out / "best_theta.npy", best["theta"])
    write_json(out / "RESULT.json", payload)
    write_csv(out / "trajectory.csv", history)
    return payload


def stage2_tier(row):
    r = float(row["final_r"])
    if r <= 1e-10:
        return 0
    if r <= 1e-8:
        return 1
    if r <= 1e-6:
        return 2
    return 3


def select_stage2_representative(rows):
    return min(rows, key=lambda r: (stage2_tier(r), int(r["total_two_qubit_gates"]), int(r["native_depth"]), int(r["actual_P"]), float(r["final_r"])))


def run_stage2():
    stage1_sel = json.loads((CAMPAIGN / "05_helmholtz_stage1" / "STAGE1_SELECTION_DECISION.json").read_text(encoding="utf-8"))["selected"]
    Ahat = np.load(Path(stage1_sel["result_directory"]) / "Ahat.npy")
    bundle = load_direct64_bundle("helmholtz_control")
    z, *_ = np.linalg.lstsq(Ahat, bundle["b_p"], rcond=None)
    learned_coeff = raw_coefficients(bundle["Dc"], z)
    out_root = CAMPAIGN / "06_helmholtz_stage2"
    out_root.mkdir(parents=True, exist_ok=True)
    families = ["vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot", "sequential_sg_mps", "dense_pairwise_ry_cnot"]
    all_rows = []
    winners = []
    for family in families:
        fam_rows = []
        targets = [1200, 1800, 2400, 3000]
        for target in targets:
            row = run_stage2_attempt(bundle, Ahat, learned_coeff, family, target, out_root / family / "P{}".format(target))
            fam_rows.append(row)
            all_rows.append(row)
            write_csv(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.csv", all_rows)
            write_json(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.json", {"rows": all_rows})
        if all(float(r["final_r"]) > 1e-10 for r in fam_rows):
            row = run_stage2_attempt(bundle, Ahat, learned_coeff, family, 4200, out_root / family / "P4200")
            fam_rows.append(row)
            all_rows.append(row)
        if all(float(r["final_r"]) > 1e-10 for r in fam_rows):
            row = run_stage2_attempt(bundle, Ahat, learned_coeff, family, 6000, out_root / family / "P6000")
            fam_rows.append(row)
            all_rows.append(row)
        winner = select_stage2_representative(fam_rows)
        winners.append(winner)
        write_json(out_root / family / "FAMILY_WINNER.json", {"winner": winner, "rows": fam_rows})
        write_csv(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.csv", all_rows)
        write_json(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.json", {"rows": all_rows})
    global_winner = select_stage2_representative(winners)
    fields = ["family", "target_P", "actual_P", "layers_or_repetitions", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "initial_r", "final_r", "PDE_L2", "u_StageII_vs_u_Ahat_LS_relative_L2", "runtime_seconds", "termination_message"]
    write_csv(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.csv", all_rows)
    write_json(out_root / "HELMHOLTZ_STAGE2_DEPTH_RESULTS.json", {"rows": all_rows})
    write_csv(out_root / "HELMHOLTZ_STAGE2_FAMILY_WINNERS.csv", winners)
    write_json(out_root / "HELMHOLTZ_STAGE2_FAMILY_WINNERS.json", {"rows": winners})
    (out_root / "HELMHOLTZ_STAGE2_REPORT.md").write_text("# Helmholtz Stage-II Depth Results\n\n" + md_table(all_rows, fields) + "\n\n## Family Winners\n\n" + md_table(winners, fields) + "\n", encoding="utf-8")
    write_json(out_root / "STAGE2_SELECTION_DECISION.json", {"selected": global_winner, "rule": "best residual tier, then lowest total two-qubit gates, native depth, P, final_r"})
    (out_root / "STAGE2_SELECTION_DECISION.md").write_text("# Stage-II Selection Decision\n\n" + md_table([global_winner], fields + ["result_directory"]) + "\n", encoding="utf-8")
    return {"rows": all_rows, "winners": winners, "selected": global_winner}


def run_stage1():
    out_root = CAMPAIGN / "05_helmholtz_stage1"
    out_root.mkdir(parents=True, exist_ok=True)
    batches = [
        ("H1-4000", [
            ("FABLE-inspired Gray", "gray_multiplexed_ry_cnot", "P4096", 4096),
            ("Block11", "vcbe_block11_real", "P3997", 3997),
            ("Block15", "vcbe_block15_real", "P3997", 3997),
        ]),
        ("H1-5000", [
            ("FABLE-inspired Gray", "gray_multiplexed_ry_cnot", "P5120", 5120),
            ("Block11", "vcbe_block11_real", "P5005", 5005),
            ("Block15", "vcbe_block15_real", "P5005", 5005),
        ]),
        ("H1-6000", [
            ("FABLE-inspired Gray", "gray_multiplexed_ry_cnot", "P6144", 6144),
            ("Block11", "vcbe_block11_real", "P6013", 6013),
            ("Block15", "vcbe_block15_real", "P5999", 5999),
        ]),
    ]
    rows = []
    for batch_name, items in batches:
        jobs = []
        for public_name, family, capacity, target_p in items:
            jobs.append({"batch": batch_name, "public_name": public_name, "family": family, "capacity": capacity, "target_P": target_p, "out_dir": str(out_root / batch_name / family / capacity)})
        with ProcessPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(stage1_job, job) for job in jobs]
            for fut in as_completed(futures):
                rows.append(fut.result())
                write_csv(out_root / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.csv", rows)
                write_json(out_root / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.json", {"rows": rows})
        batch_rows = [r for r in rows if r.get("batch") == batch_name]
        batch_rows = sorted(batch_rows, key=lambda r: (r["family"], r["actual_P"]))
        write_stage1_batch_report(out_root / batch_name, batch_name, batch_rows)
    rows = sorted(rows, key=lambda r: (r["batch"] if "batch" in r else "", r["family"], r["actual_P"]))
    selected = select_stage1(rows)
    write_csv(out_root / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.csv", rows)
    write_json(out_root / "HELMHOLTZ_STAGE1_CROSS_ARCH_RESULTS.json", {"rows": rows})
    fields = ["public_name", "capacity", "actual_P", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "e_F", "e_2", "learned_LS_PDE_L2", "learned_LS_residual", "runtime_seconds"]
    (out_root / "HELMHOLTZ_STAGE1_CROSS_ARCH_REPORT.md").write_text("# Helmholtz Stage-I Cross-Architecture Results\n\n" + md_table(rows, fields) + "\n", encoding="utf-8")
    write_json(out_root / "STAGE1_SELECTION_DECISION.json", {"selected": selected, "rule": "best residual tier, then lowest total two-qubit gates, native depth, P, max(e_F,e_2)"})
    (out_root / "STAGE1_SELECTION_DECISION.md").write_text("# Stage-I Selection Decision\n\n" + md_table([selected], fields + ["result_directory"]) + "\n", encoding="utf-8")
    return {"rows": rows, "selected": selected}


def main(argv=None):
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] == "stage1":
        result = run_stage1()
        print(json.dumps({"status": "stage1_complete", "selected": result["selected"]}, indent=2))
        return 0
    if argv[0] == "stage2":
        result = run_stage2()
        print(json.dumps({"status": "stage2_complete", "selected": result["selected"]}, indent=2))
        return 0
    raise SystemExit("unknown command {}".format(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main())
