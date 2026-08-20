"""Continue a Stage-II learned-Ahat run from an existing theta file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refined_twostage.io.artifacts import json_safe
from refined_twostage.io.hashing import short_hash_array
from refined_twostage.metrics.pde_metrics import evaluate_physical
from refined_twostage.rfm.assemble import build_rfm_system
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.objectives import objective_value_and_grad_theta, scalar_eliminated_components
from refined_twostage.stage2.polynomial import nearest_legal_capacity


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def status(path: Path, **payload) -> None:
    payload.setdefault("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(path, payload)


def resolve_stage2(family: str, target_p: int) -> dict:
    if family == "alternating_ry_cnot":
        if int(target_p) % 6:
            raise ValueError("alternating_ry_cnot target P must be divisible by 6")
        return {"family": family, "target_P": int(target_p), "layers_or_repetitions": int(target_p) // 6 - 1, "actual_P": int(target_p)}
    return nearest_legal_capacity(family, int(target_p))


def resource_row(ansatz) -> dict:
    r = ansatz.resource_counts()
    return {
        "ry_gates": int(getattr(r, "ry_count", getattr(r, "ry_gates", 0))),
        "CNOT": int(getattr(r, "cnot_count", getattr(r, "cnot_gates", 0))),
        "CZ": int(getattr(r, "cz_count", 0)),
        "total_two_qubit_gates": int(getattr(r, "two_qubit_gate_count", getattr(r, "total_two_qubit_gates", 0))),
        "native_depth": int(getattr(r, "native_depth", getattr(r, "greedy_native_critical_path_depth", 0))),
    }


def learned_stage2_diagnostics(bundle, Ahat: np.ndarray, ansatz, theta: np.ndarray, learned_coeff: np.ndarray) -> dict:
    y = ansatz.state(theta)
    comp = scalar_eliminated_components(Ahat, bundle.b_p, y)
    z = np.asarray(comp["z"], dtype=float)
    coeff = bundle.raw_coefficients(z)
    phys = evaluate_physical(bundle.pde, bundle.feature_bank, coeff, bundle.dense_x)
    learned_values = bundle.feature_bank.values(bundle.dense_x) @ learned_coeff
    stage2_values = bundle.feature_bank.values(bundle.dense_x) @ coeff
    bnorm = max(float(np.linalg.norm(bundle.b_p)), 1e-300)
    return {
        "objective_value_check": float(np.linalg.norm(comp["r_vec"]) / bnorm),
        "alpha_star": float(comp["alpha"]),
        "exact_matrix_residual": float(np.linalg.norm(bundle.A64 @ z - bundle.b_p) / bnorm),
        "learned_matrix_residual": float(np.linalg.norm(Ahat @ z - bundle.b_p) / bnorm),
        "PDE_L2": phys["PDE_relative_L2"],
        "PDE_Linf": phys["PDE_relative_Linf"],
        "dense_PDE_residual": phys["dense_PDE_residual"],
        "boundary_error": phys["boundary_error"],
        "u_StageII_vs_u_Ahat_LS_relative_L2": float(
            np.linalg.norm(stage2_values - learned_values) / max(float(np.linalg.norm(learned_values)), 1e-300)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--pde", required=True, choices=["poisson", "reaction_diffusion", "convection_diffusion"])
    parser.add_argument("--source-stage2-dir", default=None, help="Existing Stage-II directory to continue from final_theta.npy. Omit to start from a fresh theta0.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--family", default="alternating_ry_cnot")
    parser.add_argument("--target-p", type=int, default=2400)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--feature-seed", type=int, default=7061015)
    parser.add_argument("--feature-alpha", type=float, default=50.0)
    parser.add_argument("--dense-points", type=int, default=4096)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument("--maxfun", type=int, default=60000)
    parser.add_argument("--maxcor", type=int, default=100)
    parser.add_argument("--maxls", type=int, default=150)
    parser.add_argument("--ftol", type=float, default=0.0)
    parser.add_argument("--gtol", type=float, default=1e-14)
    parser.add_argument("--history-interval", type=int, default=25)
    args = parser.parse_args()

    phase = Path(args.phase)
    source_dir = Path(args.source_stage2_dir) if args.source_stage2_dir else None
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stage1_dir = phase / args.pde / "stage1_block11_M143"
    Ahat = np.load(stage1_dir / "Ahat.npy")
    bundle = build_rfm_system(args.pde, args.feature_count, args.feature_seed, args.feature_alpha, args.dense_points, canonicalize_svd=False)
    resolved = resolve_stage2(args.family, args.target_p)
    ansatz = make_stage2_ansatz(args.family, layers=int(resolved["layers_or_repetitions"]))
    if source_dir is None:
        theta0 = np.random.default_rng(13379).normal(scale=0.70, size=int(ansatz.num_parameters)).astype(float)
        start_label = "fresh theta0"
        continuation_from = ""
    else:
        theta0 = np.load(source_dir / "final_theta.npy")
        start_label = "source final_theta.npy"
        continuation_from = str(source_dir / "final_theta.npy")
    if int(ansatz.num_parameters) != int(theta0.size):
        raise ValueError(f"theta size {theta0.size} does not match ansatz parameters {ansatz.num_parameters}")

    z_ls, *_ = np.linalg.lstsq(Ahat, bundle.b_p, rcond=None)
    learned_coeff = bundle.raw_coefficients(z_ls)
    initial_value, initial_grad, _ = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta0, objective="direct_r")
    np.save(out / "theta_start.npy", theta0)
    write_json(
        out / "config.json",
        {
            "pde": args.pde,
            "source_stage2_dir": str(source_dir) if source_dir is not None else "",
            "family": args.family,
            "target_P": int(args.target_p),
            "resolved": resolved,
            "operator": "Stage-I learned Ahat",
            "objective_name": "direct_r",
            "start": start_label,
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxcor": int(args.maxcor),
            "maxls": int(args.maxls),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
        },
    )
    status(out / "progress_status.json", pde=args.pde, stage="stage2_continuation", state="running", evaluation=0, initial_r=float(initial_value))

    history: list[dict] = []
    best = {"value": float(initial_value), "theta": theta0.copy()}
    evals = 0
    t0 = time.time()

    def fun(theta):
        nonlocal evals, best
        evals += 1
        value, grad, _comp = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta, objective="direct_r")
        if value < best["value"]:
            best = {"value": float(value), "theta": np.asarray(theta, dtype=float).copy()}
        if evals == 1 or evals % int(args.history_interval) == 0:
            rec = {
                "evaluation": int(evals),
                "direct_r": float(value),
                "gradient_norm": float(np.linalg.norm(grad)),
                "best_direct_r": float(best["value"]),
                "elapsed_seconds": float(time.time() - t0),
            }
            history.append(rec)
            np.save(out / "latest_theta_checkpoint.npy", np.asarray(theta, dtype=float))
            np.save(out / "best_theta_checkpoint.npy", best["theta"])
            write_csv(out / "trajectory.csv", history)
            status(out / "progress_status.json", pde=args.pde, stage="stage2_continuation", state="running", **rec)
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxcor": int(args.maxcor),
            "maxls": int(args.maxls),
            "ftol": float(args.ftol),
            "gtol": float(args.gtol),
            "disp": False,
        },
    )
    theta = np.asarray(result.x, dtype=float)
    final_value, final_grad, _ = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta, objective="direct_r")
    diag = learned_stage2_diagnostics(bundle, Ahat, ansatz, theta, learned_coeff)
    payload = {
        "pde": args.pde,
        "family": args.family,
        "target_P": int(resolved["target_P"]),
        "actual_P": int(ansatz.num_parameters),
        "layers_or_repetitions": int(resolved["layers_or_repetitions"]),
        "objective_name": "direct_r",
        "operator": "Stage-I learned Ahat",
        "continuation_from": continuation_from,
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
        "theta_start_hash": short_hash_array(theta0),
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
    status(out / "progress_status.json", pde=args.pde, stage="stage2_continuation", state="complete", result_path=str(out / "RESULT.json"))
    print(f"result={out / 'RESULT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
