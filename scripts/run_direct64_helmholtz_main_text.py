"""Direct64 Helmholtz main-text architecture comparison.

This runner uses only migrated real-circuit implementations. Stage I runs a
dynamic queue with at most three workers. Stage II runs one worker per family;
within each family capacities are tried in increasing order using fresh
direct-r only.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refined_twostage.io.artifacts import json_safe, md_table, write_json
from refined_twostage.io.hashing import sha256_array, short_hash_array
from refined_twostage.metrics.diagnostics import classical_diagnostics
from refined_twostage.metrics.matrix_metrics import singular_diagnostics
from refined_twostage.metrics.pde_metrics import evaluate_physical
from refined_twostage.rfm.assemble import build_rfm_system
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.objectives import objective_value_and_grad_theta, scalar_eliminated_components
from refined_twostage.stage2.polynomial import nearest_legal_capacity


CAMPAIGN = ROOT / "results" / "migrated_0819_real_circuit_reproduction_20260821_014129"
PHASE = CAMPAIGN / "06_direct64_helmholtz_main_text"
PDE_KEY = "helmholtz_control"

STAGE1_JOBS = [
    {"family": "gray_multiplexed_ry_cnot", "capacity_label": "~4000", "layers_or_repetitions": 16, "actual_P": 4096},
    {"family": "gray_multiplexed_ry_cnot", "capacity_label": "~5000", "layers_or_repetitions": 20, "actual_P": 5120},
    {"family": "gray_multiplexed_ry_cnot", "capacity_label": "~6000", "layers_or_repetitions": 24, "actual_P": 6144},
    {"family": "vcbe_block11_real", "capacity_label": "~4000", "layers_or_repetitions": 95, "actual_P": 3997},
    {"family": "vcbe_block11_real", "capacity_label": "~5000", "layers_or_repetitions": 119, "actual_P": 5005},
    {"family": "vcbe_block11_real", "capacity_label": "~6000", "layers_or_repetitions": 143, "actual_P": 6013},
    {"family": "vcbe_block15_real", "capacity_label": "~4000", "layers_or_repetitions": 285, "actual_P": 3997},
    {"family": "vcbe_block15_real", "capacity_label": "~5000", "layers_or_repetitions": 357, "actual_P": 5005},
    {"family": "vcbe_block15_real", "capacity_label": "~6000", "layers_or_repetitions": 428, "actual_P": 5999},
]

STAGE2_FAMILIES = ["vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot", "sequential_sg_mps", "dense_pairwise_ry_cnot"]
STAGE2_TARGETS = [1200, 1800, 2400, 3000]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fields})


def safe_status(path: Path, **payload) -> None:
    payload.setdefault("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(path, payload)


def bundle():
    return build_rfm_system(PDE_KEY, n_features=64, seed=7061015, alpha=50.0, dense_points=4096, canonicalize_svd=False)


def stage1_dir(phase: Path, family: str, actual_p: int) -> Path:
    return phase / "s1" / family / ("P{}".format(int(actual_p)))


def stage2_dir(phase: Path, family: str, actual_p: int) -> Path:
    return phase / "s2" / family / ("P{}".format(int(actual_p)))


def stage1_resource_row(ansatz, family: str) -> dict:
    r = ansatz.resource_counts()
    if family == "gray_multiplexed_ry_cnot":
        native_depth = 32 * int(ansatz.layers) + 1
    else:
        native_depth = int(getattr(r, "native_depth", 0))
    return {
        "qubits": int(getattr(ansatz, "num_qubits", 7)),
        "one_qubit_gates": int(getattr(r, "ry_gates", 0)),
        "CNOT": int(getattr(r, "cnot_gates", 0)),
        "CZ": int(getattr(r, "cz_gates", 0)),
        "total_two_qubit_gates": int(getattr(r, "total_two_qubit_gates", getattr(r, "two_qubit_gate_count", 0))),
        "native_depth": int(native_depth),
    }


def stage2_resource_row(ansatz) -> dict:
    r = ansatz.resource_counts()
    return {
        "qubits": int(getattr(ansatz, "data_qubits", 6)),
        "one_qubit_gates": int(getattr(r, "ry_count", getattr(r, "ry_gates", 0))),
        "CNOT": int(getattr(r, "cnot_count", getattr(r, "cnot_gates", 0))),
        "CZ": int(getattr(r, "cz_count", 0)),
        "total_two_qubit_gates": int(getattr(r, "two_qubit_gate_count", getattr(r, "total_two_qubit_gates", 0))),
        "native_depth": int(getattr(r, "native_depth", getattr(r, "greedy_native_critical_path_depth", 0))),
    }


def run_stage1_job(job: dict, phase_text: str) -> dict:
    phase = Path(phase_text)
    out = stage1_dir(phase, job["family"], job["actual_P"])
    result_path = out / "result.json"
    if result_path.exists():
        row = read_json(result_path)
        row["optimizer_status"] = row.get("optimizer_status", "reused_complete")
        return row
    out.mkdir(parents=True, exist_ok=True)
    safe_status(out / "progress_status.json", state="running", stage="stage1", job=job)
    b = bundle()
    ansatz = make_stage1_ansatz(job["family"], repetitions=int(job["layers_or_repetitions"]))
    t0 = time.time()
    result = adam_optimize(
        ansatz,
        b.As,
        seed=2026072202,
        init_scale=0.02,
        steps=1000,
        lr=0.04,
        final_multiplier=0.12,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        chunk_size=8,
        history_interval=25,
    )
    initial_loss, initial_grad = ansatz.loss_and_grad_full_basis(result["theta0"], b.As, chunk_size=8, evaluator_mode="full_batch_tape")
    K = result["K"]
    Ahat = b.s_A * K
    z, coeff, ls, _ = classical_diagnostics("learned_operator", b.construction, Ahat, b.b_p, b.raw_coefficients, b.pde, b.feature_bank, b.dense_x)
    mat = singular_diagnostics(Ahat)
    np.save(out / "theta0.npy", result["theta0"])
    np.save(out / "theta_best.npy", result["theta_best"])
    np.save(out / "theta_last.npy", result["theta_last"])
    np.save(out / "theta_final.npy", result["theta"])
    np.save(out / "K_theta.npy", K)
    np.save(out / "Ahat.npy", Ahat)
    np.save(out / "target_As.npy", b.As)
    np.save(out / "learned_ls_z.npy", z)
    np.save(out / "learned_ls_coefficients.npy", coeff)
    write_csv(out / "training_history.csv", result["history"])
    row = {
        "pde": PDE_KEY,
        "data_route": "DIRECT64_REGENERATED",
        "family": job["family"],
        "capacity_label": job["capacity_label"],
        "layers_or_repetitions": int(job["layers_or_repetitions"]),
        "actual_P": int(ansatz.num_parameters),
        "seed": 2026072202,
        "init_scale": 0.02,
        "steps": 1000,
        "lr": 0.04,
        "final_lr_ratio": 0.12,
        "initial_loss": float(initial_loss),
        "initial_gradient_norm": float(np.linalg.norm(initial_grad)),
        "best_loss": float(result["best_loss"]),
        "last_loss": float(result["last_loss"]),
        "best_step": int(result["best_step"]),
        "last_step": int(result["last_step"]),
        "e_F": float(result["metrics"]["e_F"]),
        "e_2": float(result["metrics"]["e_2"]),
        "learned_LS_algebraic_residual": float(ls["classical_residual"]),
        "learned_LS_PDE_L2": float(ls["PDE_L2"]),
        "learned_LS_PDE_Linf": float(ls["PDE_Linf"]),
        "learned_LS_dense_PDE_residual": float(ls["dense_PDE_residual"]),
        "learned_LS_boundary_error": float(ls["boundary_error"]),
        "Ahat_rank@1e-12": int(mat["ranks"]["rank@1e-12"]),
        "Ahat_cond2": float(mat["cond2"]),
        "Ahat_kappa_eff@1e-12": float(mat["effective_condition_numbers"]["kappa_eff@1e-12"]),
        "runtime_seconds": float(time.time() - t0),
        "theta_hash": sha256_array(result["theta"]),
        "Ahat_hash": sha256_array(Ahat),
        "target_As_hash": sha256_array(b.As),
        "theta_final_policy": result["theta_final_policy"],
        "optimizer_status": "complete",
        "result_directory": str(out.resolve()),
    }
    row.update(stage1_resource_row(ansatz, job["family"]))
    write_json(result_path, row)
    (out / "RUN_REPORT.md").write_text("# RUN REPORT\n\n" + "\n".join("- {}: {}".format(k, v) for k, v in row.items()) + "\n", encoding="utf-8")
    safe_status(out / "progress_status.json", state="complete", stage="stage1", result_path=str(result_path))
    return row


def residual_tier(row: dict) -> int:
    value = max(float(row["e_F"]), float(row["e_2"]))
    if value <= 1e-12:
        return 0
    if value <= 1e-10:
        return 1
    if value <= 1e-8:
        return 2
    return 3


def select_stage1(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda r: (
            residual_tier(r),
            int(r["total_two_qubit_gates"]),
            int(r["native_depth"]),
            int(r["actual_P"]),
            max(float(r["e_F"]), float(r["e_2"])),
        ),
    )[0]


def run_stage1_queue(phase: Path) -> list[dict]:
    rows: list[dict] = []
    pending = list(STAGE1_JOBS)
    active = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        while pending and len(active) < 3:
            job = pending.pop(0)
            active[pool.submit(run_stage1_job, job, str(phase))] = job
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"state": "failed", "job": job, "error": str(exc), "traceback": traceback.format_exc()}
                rows.append(row)
                write_json(phase / "STAGE1_PARTIAL_RESULTS.json", {"rows": rows})
                write_csv(phase / "STAGE1_PARTIAL_RESULTS.csv", rows)
                if pending:
                    next_job = pending.pop(0)
                    active[pool.submit(run_stage1_job, next_job, str(phase))] = next_job
    if any(row.get("state") == "failed" for row in rows):
        write_json(phase / "STAGE1_RESULTS.json", {"status": "failed", "rows": rows})
        raise RuntimeError("one or more Stage-I jobs failed")
    rows = sorted(rows, key=lambda r: (r["family"], int(r["actual_P"])))
    write_json(phase / "STAGE1_RESULTS.json", {"status": "complete", "rows": rows})
    write_csv(phase / "STAGE1_RESULTS.csv", rows)
    selected = select_stage1(rows)
    write_json(phase / "STAGE1_SELECTION.json", {"selection_rule": "best residual tier, then lowest total two-qubit gates, native depth, P, max(e_F,e_2); no PDE-error selection", "selected": selected})
    src = Path(selected["result_directory"]) / "Ahat.npy"
    dst_dir = phase / "selected_stage1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "Ahat.npy")
    shutil.copy2(Path(selected["result_directory"]) / "theta_final.npy", dst_dir / "theta_final.npy")
    write_json(dst_dir / "SELECTED_STAGE1.json", selected)
    return rows


def stage2_capacity(family: str, target: int) -> dict:
    if family in ("vqls_ry_cz", "alternating_ry_cnot", "ring_ry_cnot"):
        actual = int(target)
        if actual % 6:
            actual = 6 * round(actual / 6)
        return {"family": family, "target_P": int(target), "layers_or_repetitions": actual // 6 - 1, "actual_P": actual}
    return nearest_legal_capacity(family, int(target))


def run_stage2_config(family: str, target: int, phase: Path, b, Ahat: np.ndarray) -> dict:
    cap = stage2_capacity(family, target)
    out = stage2_dir(phase, family, cap["actual_P"])
    result_path = out / "RESULT.json"
    if result_path.exists():
        return read_json(result_path)
    out.mkdir(parents=True, exist_ok=True)
    safe_status(out / "progress_status.json", state="running", stage="stage2", family=family, target_P=target)
    ansatz = make_stage2_ansatz(family, layers=int(cap["layers_or_repetitions"]))
    theta0 = np.random.default_rng(13379).normal(scale=0.70, size=int(ansatz.num_parameters)).astype(float)
    np.save(out / "theta0.npy", theta0)
    history: list[dict] = []
    best = {"value": float("inf"), "theta": theta0.copy()}
    evals = [0]
    t0 = time.time()
    initial_r, initial_grad, _ = objective_value_and_grad_theta(Ahat, b.b_p, ansatz, theta0, objective="direct_r")

    def objective(theta):
        evals[0] += 1
        value, grad, _ = objective_value_and_grad_theta(Ahat, b.b_p, ansatz, theta, objective="direct_r")
        if value < best["value"]:
            best["value"] = float(value)
            best["theta"] = np.asarray(theta, dtype=float).copy()
        if evals[0] == 1 or evals[0] % 25 == 0:
            rec = {
                "evaluation": int(evals[0]),
                "direct_r": float(value),
                "gradient_norm": float(np.linalg.norm(grad)),
                "best_direct_r": float(best["value"]),
                "elapsed_seconds": float(time.time() - t0),
            }
            history.append(rec)
            write_csv(out / "trajectory.csv", history)
            safe_status(out / "progress_status.json", state="running", stage="stage2", family=family, **rec)
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 30000, "maxfun": 60000, "maxcor": 100, "maxls": 150, "ftol": 1e-16, "gtol": 1e-12, "disp": False},
    )
    terminal_theta = np.asarray(result.x, dtype=float)
    selected_theta = best["theta"] if best["value"] <= float(result.fun) else terminal_theta
    selected_r = float(best["value"] if best["value"] <= float(result.fun) else result.fun)
    np.save(out / "final_theta.npy", terminal_theta)
    np.save(out / "best_theta.npy", selected_theta)
    y = ansatz.state(selected_theta)
    comp = scalar_eliminated_components(Ahat, b.b_p, y)
    z = np.asarray(comp["z"], dtype=float)
    coeff = b.raw_coefficients(z)
    phys = evaluate_physical(b.pde, b.feature_bank, coeff, b.dense_x)
    value_check = float(np.linalg.norm(comp["r_vec"]) / max(float(np.linalg.norm(b.b_p)), 1e-300))
    _final_value, final_grad, _ = objective_value_and_grad_theta(Ahat, b.b_p, ansatz, selected_theta, objective="direct_r")
    row = {
        "pde": PDE_KEY,
        "data_route": "DIRECT64_REGENERATED",
        "family": family,
        "target_P": int(target),
        "actual_P": int(ansatz.num_parameters),
        "layers_or_repetitions": int(cap["layers_or_repetitions"]),
        "initial_r": float(initial_r),
        "final_direct_r": float(selected_r),
        "raw_algebraic_residual": float(np.linalg.norm(b.A64 @ z - b.b_p) / max(float(np.linalg.norm(b.b_p)), 1e-300)),
        "learned_matrix_residual": value_check,
        "alpha_star": float(comp["alpha"]),
        "PDE_L2": float(phys["PDE_relative_L2"]),
        "PDE_Linf": float(phys["PDE_relative_Linf"]),
        "dense_PDE_residual": float(phys["dense_PDE_residual"]),
        "boundary_error": float(phys["boundary_error"]),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(result.nfev),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "gradient_norm": float(np.linalg.norm(final_grad)),
        "runtime_seconds": float(time.time() - t0),
        "theta_hash": short_hash_array(selected_theta),
        "theta0_hash": short_hash_array(theta0),
        "result_directory": str(out.resolve()),
        "objective_value_matches_explicit_residual": bool(abs(value_check - selected_r) <= 1e-10 * max(1.0, abs(selected_r))),
    }
    row.update(stage2_resource_row(ansatz))
    write_json(result_path, row)
    write_csv(out / "trajectory.csv", history)
    (out / "RUN_REPORT.md").write_text("# RUN REPORT\n\n" + "\n".join("- {}: {}".format(k, v) for k, v in row.items()) + "\n", encoding="utf-8")
    safe_status(out / "progress_status.json", state="complete", stage="stage2", result_path=str(result_path))
    return row


def run_stage2_family(family: str, phase_text: str) -> dict:
    phase = Path(phase_text)
    b = bundle()
    Ahat = np.load(phase / "selected_stage1" / "Ahat.npy")
    rows: list[dict] = []
    success_seen = False
    confirm_run_done = False
    for target in STAGE2_TARGETS:
        row = run_stage2_config(family, target, phase, b, Ahat)
        rows.append(row)
        write_json(phase / "STAGE2_PARTIAL_{}.json".format(family), {"family": family, "rows": rows})
        if success_seen:
            confirm_run_done = True
        if float(row["final_direct_r"]) <= 1e-10:
            success_seen = True
        if success_seen and confirm_run_done:
            break
    best = sorted(rows, key=lambda r: (float(r["final_direct_r"]), int(r["total_two_qubit_gates"]), int(r["native_depth"]), int(r["actual_P"])))[0]
    operating = sorted(rows, key=lambda r: (0 if float(r["final_direct_r"]) <= max(1e-10, float(best["final_direct_r"]) * 10.0) else 1, int(r["total_two_qubit_gates"]), int(r["native_depth"]), int(r["actual_P"]), float(r["final_direct_r"])))[0]
    return {"family": family, "rows": rows, "best": best, "operating_point": operating}


def run_stage2(phase: Path) -> list[dict]:
    rows: list[dict] = []
    summaries = []
    with ProcessPoolExecutor(max_workers=len(STAGE2_FAMILIES)) as pool:
        futures = {pool.submit(run_stage2_family, family, str(phase)): family for family in STAGE2_FAMILIES}
        for future in futures:
            summary = future.result()
            summaries.append(summary)
            rows.extend(summary["rows"])
            write_json(phase / "STAGE2_PARTIAL_RESULTS.json", {"summaries": summaries, "rows": rows})
            write_csv(phase / "STAGE2_PARTIAL_RESULTS.csv", rows)
    rows = sorted(rows, key=lambda r: (r["family"], int(r["actual_P"])))
    write_json(phase / "STAGE2_RESULTS.json", {"status": "complete", "summaries": summaries, "rows": rows})
    write_csv(phase / "STAGE2_RESULTS.csv", rows)
    return rows


def write_report(phase: Path, pytest_result: str) -> Path:
    b = bundle()
    _z, _coeff, classical, _mat = b.exact_working_diagnostics()
    stage1 = read_json(phase / "STAGE1_RESULTS.json")["rows"]
    selected = read_json(phase / "STAGE1_SELECTION.json")["selected"]
    stage2_payload = read_json(phase / "STAGE2_RESULTS.json")
    stage2 = stage2_payload["rows"]
    family_rows = [summary["operating_point"] for summary in stage2_payload["summaries"]]
    final = sorted(family_rows, key=lambda r: (float(r["final_direct_r"]), int(r["total_two_qubit_gates"]), int(r["native_depth"]), int(r["actual_P"])))[0]
    lines = [
        "# Direct64 Helmholtz Main-Text Experiment Report",
        "",
        "Campaign root: `{}`".format(CAMPAIGN),
        "",
        "## 1. Benchmark Definition",
        "",
        "Problem: Direct64 Helmholtz control, 64 collocation rows and 64 sine features, no 200-feature right-SVD working-space reduction. The operator uses `Dr=I`, column scaling `Dc`, `A64=A_p`, and `s_A=1.05*||A64||_2`.",
        "",
        "## 2. Classical Baseline",
        "",
        md_table([classical], ["shape", "rank@1e-12", "cond2", "kappa_eff@1e-12", "classical_residual", "PDE_L2", "PDE_Linf", "dense_PDE_residual", "boundary_error"]),
        "",
        "## 3. Exact Stage-I Protocol",
        "",
        "All Stage-I jobs used migrated real circuits, full-basis analytic-gradient Adam, seed `2026072202`, init scale `0.02`, 1000 steps, lr `0.04`, cosine final multiplier `0.12`, beta1 `0.9`, beta2 `0.999`, eps `1e-8`, chunk size `8`, float64. No dense surrogate was used.",
        "",
        "## 4. All 9 Stage-I Results",
        "",
        md_table(stage1, ["family", "capacity_label", "layers_or_repetitions", "actual_P", "qubits", "one_qubit_gates", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "initial_loss", "best_loss", "last_loss", "best_step", "e_F", "e_2", "learned_LS_algebraic_residual", "learned_LS_PDE_L2", "learned_LS_PDE_Linf", "learned_LS_dense_PDE_residual", "learned_LS_boundary_error", "Ahat_rank@1e-12", "Ahat_cond2", "Ahat_kappa_eff@1e-12", "runtime_seconds", "theta_hash", "Ahat_hash", "optimizer_status"]),
        "",
        "## 5. Stage-I Selection",
        "",
        "Selection rule: best residual tier, then lowest total two-qubit gates, native depth, P, and max(e_F,e_2). PDE error is not used for selection.",
        "",
        md_table([selected], ["family", "capacity_label", "layers_or_repetitions", "actual_P", "total_two_qubit_gates", "native_depth", "e_F", "e_2", "learned_LS_PDE_L2", "result_directory"]),
        "",
        "The selected learned operator is frozen at `{}`.".format(str((phase / "selected_stage1" / "Ahat.npy").resolve())),
        "",
        "## 6. Exact Stage-II Fresh Direct-r Protocol",
        "",
        "All Stage-II jobs used migrated real polynomial circuits, fresh direct-r only, seed `13379`, init scale `0.70`, L-BFGS-B with maxiter `30000`, maxfun `60000`, maxcor `100`, maxls `150`, ftol `1e-16`, and gtol `1e-12`. No r^2 warm start and no historical theta warm start were used.",
        "",
        "## 7. All Stage-II Capacities Tested",
        "",
        md_table(stage2, ["family", "target_P", "layers_or_repetitions", "actual_P", "qubits", "one_qubit_gates", "CNOT", "CZ", "total_two_qubit_gates", "native_depth", "initial_r", "final_direct_r", "raw_algebraic_residual", "learned_matrix_residual", "alpha_star", "PDE_L2", "PDE_Linf", "dense_PDE_residual", "boundary_error", "nit", "nfev", "njev", "optimizer_success", "runtime_seconds", "theta_hash"]),
        "",
        "## 8. Five-Family Stage-II Operating Points",
        "",
        md_table(family_rows, ["family", "actual_P", "total_two_qubit_gates", "native_depth", "final_direct_r", "raw_algebraic_residual", "PDE_L2", "PDE_Linf", "runtime_seconds", "result_directory"]),
        "",
        "## 9. Final Manuscript Recommendation",
        "",
        md_table([final], ["family", "actual_P", "total_two_qubit_gates", "native_depth", "final_direct_r", "PDE_L2", "PDE_Linf", "result_directory"]),
        "",
        "Recommended Stage-I + Stage-II configuration: Stage-I `{}` P{} with Stage-II `{}` P{}.".format(selected["family"], selected["actual_P"], final["family"], final["actual_P"]),
        "",
        "## 10. Runtime Information",
        "",
        "Stage-I total worker runtime seconds: {:.6e}. Stage-II total worker runtime seconds: {:.6e}.".format(sum(float(r["runtime_seconds"]) for r in stage1), sum(float(r["runtime_seconds"]) for r in stage2)),
        "",
        "## 11. Failed or Unfinished Jobs",
        "",
        "No failed or unfinished Direct64 Helmholtz main-text jobs are recorded in `06_direct64_helmholtz_main_text`." if all(r.get("optimizer_success", True) for r in stage2) else "At least one Stage-II optimizer reported unsuccessful termination; inspect `STAGE2_RESULTS.json`.",
        "",
        "## 12. Exact Artifact Paths",
        "",
        md_table(stage1, ["family", "actual_P", "result_directory"]),
        "",
        md_table(stage2, ["family", "actual_P", "result_directory"]),
        "",
        "## 13. Exact Reproduction Commands",
        "",
        "```powershell",
        "python scripts\\run_direct64_helmholtz_main_text.py",
        "python -m pytest -q",
        "```",
        "",
        "## 14. Pytest",
        "",
        pytest_result,
        "",
    ]
    report = phase / "DIRECT64_HELMHOLTZ_MAIN_TEXT_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(report, CAMPAIGN / "DIRECT64_HELMHOLTZ_MAIN_TEXT_REPORT.md")
    shutil.copy2(report, CAMPAIGN / "EXPERIMENT_RESULT_REPORT.md")
    return report


def import_existing_block11_p6013(phase: Path) -> None:
    src = CAMPAIGN / "04_phaseB_direct64" / "helmholtz_control" / "stage1_block11_M143"
    dst = stage1_dir(phase, "vcbe_block11_real", 6013)
    if (dst / "result.json").exists() or not (src / "result.json").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["result.json", "RUN_REPORT.md", "theta0.npy", "theta_best.npy", "theta_last.npy", "theta_final.npy", "K_theta.npy", "Ahat.npy", "target_As.npy", "training_history.csv", "progress_status.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    row = read_json(dst / "result.json")
    row.update({"capacity_label": "~6000", "layers_or_repetitions": 143, "actual_P": 6013, "reused_from": str(src.resolve()), "optimizer_status": "reused_complete"})
    write_json(dst / "result.json", row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default=str(PHASE))
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--pytest-result", default="Not rerun by this script invocation.")
    ns = parser.parse_args()
    phase = Path(ns.phase)
    if not phase.is_absolute():
        phase = ROOT / phase
    phase.mkdir(parents=True, exist_ok=True)
    write_json(phase / "CONFIG.json", {"stage1_jobs": STAGE1_JOBS, "stage2_families": STAGE2_FAMILIES, "stage2_targets": STAGE2_TARGETS})
    _z, _coeff, classical, _mat = bundle().exact_working_diagnostics()
    write_json(phase / "CLASSICAL_BASELINE.json", classical)
    import_existing_block11_p6013(phase)
    if not ns.skip_stage1:
        run_stage1_queue(phase)
    if not ns.skip_stage2:
        run_stage2(phase)
    report = write_report(phase, ns.pytest_result)
    print("phase={}".format(phase))
    print("report={}".format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

