"""Parallel cross-PDE Stage-I -> Stage-II runner.

Each PDE owns one worker process.  A worker runs Stage-I first, then uses the
learned Stage-I operator Ahat as the Stage-II operator for that same PDE.
Existing per-stage result files are reused so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refined_twostage.io.artifacts import json_safe, project_root
from refined_twostage.io.hashing import sha256_array, short_hash_array
from refined_twostage.metrics.diagnostics import classical_diagnostics
from refined_twostage.metrics.pde_metrics import evaluate_physical
from refined_twostage.rfm.assemble import build_rfm_system
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.objectives import objective_value_and_grad_theta, scalar_eliminated_components
from refined_twostage.stage2.polynomial import nearest_legal_capacity


PDE_KEYS = ("helmholtz_control", "poisson", "reaction_diffusion", "convection_diffusion")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def status(path: Path, **payload) -> None:
    payload.setdefault("updated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(path, payload)


def resource_row(ansatz) -> dict:
    r = ansatz.resource_counts()
    return {
        "ry_gates": int(getattr(r, "ry_count", getattr(r, "ry_gates", 0))),
        "CNOT": int(getattr(r, "cnot_count", getattr(r, "cnot_gates", 0))),
        "CZ": int(getattr(r, "cz_count", 0)),
        "total_two_qubit_gates": int(getattr(r, "two_qubit_gate_count", getattr(r, "total_two_qubit_gates", 0))),
        "native_depth": int(getattr(r, "native_depth", getattr(r, "greedy_native_critical_path_depth", 0))),
    }


def resolve_stage2(family: str, target_p: int) -> dict:
    if family == "alternating_ry_cnot":
        if int(target_p) % 6:
            raise ValueError("alternating_ry_cnot target P must be divisible by 6")
        return {"family": family, "target_P": int(target_p), "layers_or_repetitions": int(target_p) // 6 - 1, "actual_P": int(target_p)}
    return nearest_legal_capacity(family, int(target_p))


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


def run_stage1(pde_key: str, phase: Path, args: dict, bundle) -> dict:
    out = phase / pde_key / f"stage1_block11_M{args['stage1_repetitions']}"
    result_path = out / "result.json"
    if result_path.exists():
        row = read_json(result_path)
        return {"row": row, "Ahat": np.load(out / "Ahat.npy"), "reused": True, "out": str(out)}

    out.mkdir(parents=True, exist_ok=True)
    status(out / "progress_status.json", pde=pde_key, stage="stage1", state="running", step="adam_optimize")
    t0 = time.time()
    ansatz = make_stage1_ansatz(args["stage1_family"], repetitions=int(args["stage1_repetitions"]))
    checkpoint_path = out / "stage1_checkpoint_latest.npz"
    resume_checkpoint = checkpoint_path if checkpoint_path.exists() else None
    result = adam_optimize(
        ansatz,
        bundle.As,
        seed=int(args["stage1_seed"]),
        init_scale=float(args["stage1_init_scale"]),
        steps=int(args["stage1_steps"]),
        lr=float(args["stage1_lr"]),
        final_multiplier=float(args["stage1_final_multiplier"]),
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        chunk_size=int(args["stage1_chunk_size"]),
        history_interval=int(args["stage1_history_interval"]),
        checkpoint_dir=out,
        checkpoint_interval=int(args["stage1_history_interval"]),
        resume_checkpoint=resume_checkpoint,
    )
    K = result["K"]
    Ahat = bundle.s_A * K
    _z, _coeff, diag, _mat = classical_diagnostics(
        "learned_operator",
        bundle.construction,
        Ahat,
        bundle.b_p,
        bundle.raw_coefficients,
        bundle.pde,
        bundle.feature_bank,
        bundle.dense_x,
    )
    np.save(out / "theta0.npy", result["theta0"])
    np.save(out / "theta_best.npy", result["theta_best"])
    np.save(out / "theta_last.npy", result["theta_last"])
    np.save(out / "theta_final.npy", result["theta"])
    np.save(out / "K_theta.npy", K)
    np.save(out / "Ahat.npy", Ahat)
    np.save(out / "target_As.npy", bundle.As)
    write_csv(out / "training_history.csv", result["history"])
    row = {
        "pde": pde_key,
        "family": args["stage1_family"],
        "M": int(args["stage1_repetitions"]),
        "P": int(ansatz.num_parameters),
        "CNOT": int(ansatz.resource_counts().cnot_gates),
        "native_depth": int(ansatz.resource_counts().native_depth),
        "seed": int(args["stage1_seed"]),
        "init_scale": float(args["stage1_init_scale"]),
        "steps": int(args["stage1_steps"]),
        "lr": float(args["stage1_lr"]),
        "final_lr_ratio": float(args["stage1_final_multiplier"]),
        "best_loss": result["best_loss"],
        "last_loss": result["last_loss"],
        "best_step": result["best_step"],
        "last_step": result["last_step"],
        "e_F": result["metrics"]["e_F"],
        "e_2": result["metrics"]["e_2"],
        "best_gradient_norm": result["best_gradient_norm"],
        "last_gradient_norm": result["last_gradient_norm"],
        "theta_final_policy": result["theta_final_policy"],
        "learned_LS_residual": diag["classical_residual"],
        "learned_LS_PDE_L2": diag["PDE_L2"],
        "learned_LS_PDE_Linf": diag["PDE_Linf"],
        "learned_LS_dense_PDE_residual": diag["dense_PDE_residual"],
        "learned_LS_boundary_error": diag["boundary_error"],
        "theta0_hash": sha256_array(result["theta0"]),
        "theta_best_hash": sha256_array(result["theta_best"]),
        "theta_last_hash": sha256_array(result["theta_last"]),
        "K_hash": sha256_array(K),
        "Ahat_hash": sha256_array(Ahat),
        "target_As_hash": sha256_array(bundle.As),
        "runtime_seconds": float(time.time() - t0),
        "run_dir": str(out),
    }
    write_json(result_path, row)
    (out / "RUN_REPORT.md").write_text("# RUN REPORT\n\n" + "\n".join(f"- {k}: {v}" for k, v in row.items()) + "\n", encoding="utf-8")
    status(out / "progress_status.json", pde=pde_key, stage="stage1", state="complete", result_path=str(result_path))
    return {"row": row, "Ahat": Ahat, "reused": False, "out": str(out)}


def run_stage2(pde_key: str, phase: Path, args: dict, bundle, Ahat: np.ndarray) -> dict:
    resolved = resolve_stage2(args["stage2_family"], int(args["stage2_target_p"]))
    out = phase / pde_key / "stage2_learned_Ahat" / args["stage2_family"] / f"P{resolved['target_P']}"
    result_path = out / "RESULT.json"
    if result_path.exists():
        return {"row": read_json(result_path), "reused": True, "out": str(out)}

    out.mkdir(parents=True, exist_ok=True)
    status(out / "progress_status.json", pde=pde_key, stage="stage2", state="running", evaluation=0)
    ansatz = make_stage2_ansatz(args["stage2_family"], layers=int(resolved["layers_or_repetitions"]))
    checkpoint_path = out / "best_theta_checkpoint.npy"
    if checkpoint_path.exists():
        theta0 = np.load(checkpoint_path).astype(float)
        start_mode = "resume_best_theta_checkpoint"
    else:
        theta0 = np.random.default_rng(int(args["stage2_seed"])).normal(scale=float(args["stage2_init_scale"]), size=int(ansatz.num_parameters)).astype(float)
        start_mode = "fresh_theta0"
    if theta0.size != int(ansatz.num_parameters):
        raise ValueError("Stage-II checkpoint theta has {} parameters; ansatz expects {}".format(theta0.size, ansatz.num_parameters))
    np.save(out / "theta0.npy", theta0)
    z_ls, *_ = np.linalg.lstsq(Ahat, bundle.b_p, rcond=None)
    learned_coeff = bundle.raw_coefficients(z_ls)
    history: list[dict] = []
    best = {"value": float("inf"), "theta": theta0.copy()}
    evals = 0
    t0 = time.time()
    initial_value, initial_grad, _ = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta0, objective="direct_r")

    def fun(theta):
        nonlocal evals, best
        evals += 1
        value, grad, _comp = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta, objective="direct_r")
        if value < best["value"]:
            best = {"value": float(value), "theta": np.asarray(theta, dtype=float).copy()}
        if evals == 1 or evals % int(args["stage2_history_interval"]) == 0:
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
            status(out / "progress_status.json", pde=pde_key, stage="stage2", state="running", **rec)
        return float(value), np.asarray(grad, dtype=float)

    write_json(
        out / "config.json",
        {
            "family": args["stage2_family"],
            "target_P": int(args["stage2_target_p"]),
            "resolved": resolved,
            "operator": "Stage-I learned Ahat",
            "start_mode": start_mode,
            "objective_name": "direct_r",
            "theta0_seed": int(args["stage2_seed"]),
            "init_scale": float(args["stage2_init_scale"]),
            "optimizer": "L-BFGS-B",
            "maxiter": int(args["stage2_maxiter"]),
            "maxfun": int(args["stage2_maxfun"]),
            "maxcor": int(args["stage2_maxcor"]),
            "maxls": int(args["stage2_maxls"]),
            "ftol": float(args["stage2_ftol"]),
            "gtol": float(args["stage2_gtol"]),
        },
    )
    result = minimize(
        fun,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(args["stage2_maxiter"]),
            "maxfun": int(args["stage2_maxfun"]),
            "maxcor": int(args["stage2_maxcor"]),
            "maxls": int(args["stage2_maxls"]),
            "ftol": float(args["stage2_ftol"]),
            "gtol": float(args["stage2_gtol"]),
            "disp": False,
        },
    )
    theta = np.asarray(result.x, dtype=float)
    final_value, final_grad, _ = objective_value_and_grad_theta(Ahat, bundle.b_p, ansatz, theta, objective="direct_r")
    diag = learned_stage2_diagnostics(bundle, Ahat, ansatz, theta, learned_coeff)
    row = {
        "pde": pde_key,
        "family": args["stage2_family"],
        "target_P": int(resolved["target_P"]),
        "actual_P": int(ansatz.num_parameters),
        "layers_or_repetitions": int(resolved["layers_or_repetitions"]),
        "objective_name": "direct_r",
        "operator": "Stage-I learned Ahat",
        "start_mode": start_mode,
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
    row.update(resource_row(ansatz))
    row.update(diag)
    row["objective_value_matches_explicit_residual"] = bool(abs(row["final_r"] - row["objective_value_check"]) <= 1e-10 * max(1.0, abs(row["final_r"])))
    np.save(out / "final_theta.npy", theta)
    np.save(out / "best_theta.npy", best["theta"])
    write_json(result_path, row)
    write_csv(out / "trajectory.csv", history)
    status(out / "progress_status.json", pde=pde_key, stage="stage2", state="complete", result_path=str(result_path))
    return {"row": row, "reused": False, "out": str(out)}


def run_pde(pde_key: str, phase: str, args: dict) -> dict:
    try:
        phase_path = Path(phase)
        status(phase_path / pde_key / "worker_status.json", pde=pde_key, state="starting")
        bundle = build_rfm_system(
            pde_key,
            int(args["feature_count"]),
            int(args["feature_seed"]),
            float(args["feature_alpha"]),
            int(args["dense_points"]),
            canonicalize_svd=bool(args["canonicalize_svd"]),
        )
        stage1 = run_stage1(pde_key, phase_path, args, bundle)
        status(phase_path / pde_key / "worker_status.json", pde=pde_key, state="stage1_complete", stage1_reused=stage1["reused"])
        stage2 = run_stage2(pde_key, phase_path, args, bundle, stage1["Ahat"])
        payload = {"pde": pde_key, "state": "complete", "stage1": stage1["row"], "stage2": stage2["row"], "stage1_reused": stage1["reused"], "stage2_reused": stage2["reused"]}
        status(phase_path / pde_key / "worker_status.json", **payload)
        return payload
    except Exception as exc:
        err = {"pde": pde_key, "state": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        status(Path(phase) / pde_key / "worker_status.json", **err)
        return err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default=None, help="Output phase directory. Defaults to results/parallel_crosspde_twostage_<timestamp>.")
    parser.add_argument("--pdes", nargs="+", default=list(PDE_KEYS), choices=PDE_KEYS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--feature-count", type=int, default=200)
    parser.add_argument("--feature-seed", type=int, default=7061015)
    parser.add_argument("--feature-alpha", type=float, default=50.0)
    parser.add_argument("--dense-points", type=int, default=4096)
    parser.add_argument("--canonicalize-svd", action="store_true", default=False)
    parser.add_argument("--stage1-family", default="vcbe_block11_real")
    parser.add_argument("--stage1-repetitions", type=int, default=143)
    parser.add_argument("--stage1-seed", type=int, default=2026072202)
    parser.add_argument("--stage1-init-scale", type=float, default=0.02)
    parser.add_argument("--stage1-steps", type=int, default=1000)
    parser.add_argument("--stage1-lr", type=float, default=0.04)
    parser.add_argument("--stage1-final-multiplier", type=float, default=0.12)
    parser.add_argument("--stage1-chunk-size", type=int, default=8)
    parser.add_argument("--stage1-history-interval", type=int, default=25)
    parser.add_argument("--stage2-family", default="alternating_ry_cnot")
    parser.add_argument("--stage2-target-p", type=int, default=2400)
    parser.add_argument("--stage2-seed", type=int, default=13379)
    parser.add_argument("--stage2-init-scale", type=float, default=0.70)
    parser.add_argument("--stage2-maxiter", type=int, default=30000)
    parser.add_argument("--stage2-maxfun", type=int, default=60000)
    parser.add_argument("--stage2-maxcor", type=int, default=100)
    parser.add_argument("--stage2-maxls", type=int, default=150)
    parser.add_argument("--stage2-ftol", type=float, default=1e-16)
    parser.add_argument("--stage2-gtol", type=float, default=1e-12)
    parser.add_argument("--stage2-history-interval", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    phase = Path(ns.phase) if ns.phase else project_root() / "results" / f"parallel_crosspde_twostage_{time.strftime('%Y%m%d_%H%M%S')}"
    if not phase.is_absolute():
        phase = project_root() / phase
    phase.mkdir(parents=True, exist_ok=True)
    args = vars(ns)
    args["phase"] = str(phase)
    write_json(phase / "parallel_config.json", args)
    status(phase / "parallel_status.json", state="running", pdes=list(ns.pdes), workers=min(int(ns.workers), len(ns.pdes)))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(int(ns.workers), len(ns.pdes))) as pool:
        futures = {pool.submit(run_pde, pde_key, str(phase), args): pde_key for pde_key in ns.pdes}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(phase / "parallel_twostage_results.json", {"rows": rows})
            flat_rows = []
            for item in rows:
                if item.get("state") == "complete":
                    flat_rows.append({"pde": item["pde"], **{f"stage1_{k}": v for k, v in item["stage1"].items()}, **{f"stage2_{k}": v for k, v in item["stage2"].items()}})
                else:
                    flat_rows.append(item)
            write_csv(phase / "parallel_twostage_results.csv", flat_rows)

    ok = all(row.get("state") == "complete" for row in rows)
    status(phase / "parallel_status.json", state="complete" if ok else "failed", result_path=str(phase / "parallel_twostage_results.json"))
    print(f"phase={phase}")
    print("status={}".format("complete" if ok else "failed"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
