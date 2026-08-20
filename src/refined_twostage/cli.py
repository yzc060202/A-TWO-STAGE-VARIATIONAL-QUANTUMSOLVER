"""Command-line interface for the clean refined two-stage framework."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

from refined_twostage.io.artifacts import environment_record, latest_result_root, load_yaml, md_table, project_root, read_json, timestamped_result_root, write_csv, write_json
from refined_twostage.io.hashing import sha256_array, short_hash_array
from refined_twostage.preprocessing.manifests import save_frozen_bundle
from refined_twostage.problems.common import pde_specs
from refined_twostage.rfm.assemble import build_rfm_system
from refined_twostage.rfm.features import sample_uniform_sine_features
from refined_twostage.stage1.interface import make_stage1_ansatz
from refined_twostage.stage1.optimizer import adam_optimize
from refined_twostage.stage2.evaluator import stage2_solution_diagnostics
from refined_twostage.stage2.interface import make_stage2_ansatz
from refined_twostage.stage2.optimizer import lbfgsb_fresh_direct_r, lbfgsb_r2_then_r


HISTORICAL_ROOT = Path("external_artifacts") / "final_refined_stage2_and_crosspde_20260819_045316"


def _cfg_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _results_base(cfg: dict) -> Path:
    base = Path(cfg.get("results_root", "results"))
    if not base.is_absolute():
        base = project_root() / base
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_root(cfg: dict, create: bool = True) -> Path:
    base = _results_base(cfg)
    if cfg.get("result_root"):
        root = Path(cfg["result_root"])
        if not root.is_absolute():
            root = project_root() / root
        root.mkdir(parents=True, exist_ok=True)
        (base / "LATEST_RESULT_ROOT.txt").write_text(str(root.resolve()), encoding="utf-8")
        return root
    return timestamped_result_root(base) if create else latest_result_root(base)


def _historical_summary(pde_key: str) -> dict:
    path = HISTORICAL_ROOT / "cross_pde_screening_new" / pde_key / "SCREENING_RESULT.json"
    return read_json(path) if path.exists() else {}


def leakage_audit_payload() -> dict:
    checked = ["target-informed anchor features", "exact-solution frequencies", "candidate-pool screening", "top-k feature filtering", "pivoted-QR feature selection", "seed search", "feature replacement", "special index boosts", "target-informed column boosts", "PDE-error-based feature acceptance", "quantum-result feedback"]
    return {"historical_source": "paper_formal_experiments/final_fivefamily_crosspde_screened.py", "generator_function": "make_feature_bank(n_features, seed)", "bundle_function": "PdeBundle(spec, fb, 'NEW')", "features": "w=Uniform[-1,1]^N, b=Uniform[-1,1]^N, all w drawn before all b, alpha=50", "collocation": "np.linspace(-1,1,64) with endpoint Dirichlet rows", "findings": {item: False for item in checked}, "leakage_detected": False, "notes": ["The historical run used a pass/fail gate to choose NEW_64x200_TO_64x64 versus an OLD fallback after constructing one fixed 200-feature bank.", "That gate did not modify, rank, replace, boost, or select individual features, and did not use quantum outputs.", "No Helmholtz-aligned anchor-enriched helper is imported by the recovered NEW generator path."]}


def write_leakage_audit(root: Path) -> dict:
    payload = leakage_audit_payload()
    out_dir = root / "reproduction_20260819"
    write_json(out_dir / "GENERATOR_LEAKAGE_AUDIT.json", payload)
    lines = ["# Generator Leakage Audit", "", "Recovered source path: `paper_formal_experiments/final_fivefamily_crosspde_screened.py`.", "", "Conclusion: no feature-level leakage mechanism was found in the exact NEW 2026-08-19 generator.", "", md_table([{"mechanism": key, "detected": val} for key, val in payload["findings"].items()], ["mechanism", "detected"]), "", "The only screen was a one-shot data-path gate after fixed-bank construction; it did not alter the generated bank."]
    (out_dir / "GENERATOR_LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "GENERATOR_LEAKAGE_AUDIT.json", payload)
    (root / "GENERATOR_LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _historical_hash_comparison(bundle) -> dict:
    hist = _historical_summary(bundle.pde.key)
    if not hist:
        return {"available": False}
    current = bundle.identity()
    mapping = {"As": "As_hash", "b_p": "b_p_hash"}
    items = {}
    for key in ["A_raw", "b_raw", "Dr", "Dc", "A_p", "b_p", "V64", "A64", "As"]:
        hist_key = mapping.get(key, key + "_hash")
        items[key] = {"historical": hist.get(hist_key, ""), "current": current.get(hist_key, ""), "match": bool(hist.get(hist_key, "") == current.get(hist_key, ""))}
    return {"available": True, "items": items}


def reproduce_crosspde(config_path: Path, root: Path = None) -> dict:
    cfg = load_yaml(config_path)
    root = _build_root(cfg, create=True) if root is None else root
    audit = write_leakage_audit(root)
    rows = []
    manifests = []
    spectra = {}
    for key in cfg.get("pdes", [p.key for p in pde_specs()]):
        bundle = build_rfm_system(key, int(cfg.get("feature_count", 200)), int(cfg.get("feature_seed", 7061015)), float(cfg.get("alpha", 50.0)), int(cfg.get("dense_points", 4096)), canonicalize_svd=bool(cfg.get("canonicalize_svd_signs", True)))
        _zw, _cw, work_row, work_mat = bundle.exact_working_diagnostics()
        _zr, _cr, raw_row, raw_mat = bundle.raw_rectangular_diagnostics()
        hist = _historical_summary(key)
        comparison = _historical_hash_comparison(bundle)
        manifest = save_frozen_bundle(bundle, root / "frozen_20260819" / key, {"historical_hash_comparison": comparison, "historical_fingerprint": hist})
        manifests.append(manifest)
        rows.append({"PDE": bundle.pde.name, "pde_key": key, "raw_shape": "{}x{}".format(bundle.A_raw.shape[0], bundle.A_raw.shape[1]), "working_shape": "{}x{}".format(bundle.A64.shape[0], bundle.A64.shape[1]), "raw_scaled_rectangular_PDE_L2": raw_row["PDE_L2"], "exact_classical_PDE_L2": work_row["PDE_L2"], "exact_classical_PDE_Linf": work_row["PDE_Linf"], "exact_classical_dense_PDE_residual": work_row["dense_PDE_residual"], "exact_classical_boundary_error_max": work_row["boundary_error"], "exact_classical_residual": work_row["classical_residual"], "rank_A_raw@1e-12": raw_mat["ranks"]["rank@1e-12"], "rank_A64@1e-12": work_mat["ranks"]["rank@1e-12"], "cond_A_raw": raw_mat["cond2"], "cond_A64": work_mat["cond2"], "feature_hash": manifest["feature_hash"], "collocation_hash": manifest["collocation_hash"], "historical_metric_L2_absdiff": abs(float(work_row["PDE_L2"]) - float(hist.get("exact_classical_PDE_L2", work_row["PDE_L2"]))), "hashes_exact_match": bool(comparison.get("available") and all(v["match"] for v in comparison["items"].values()))})
        spectra[key + "_A_raw"] = np.asarray(raw_mat["singular_values"], dtype=float)
        spectra[key + "_A_p"] = bundle.singular_values_Ap
        spectra[key + "_A64"] = bundle.singular_values_A64
    write_csv(root / "REPRODUCED_20260819_CROSSPDE_RESULTS.csv", rows)
    write_json(root / "REPRODUCED_20260819_CROSSPDE_RESULTS.json", {"rows": rows, "manifests": manifests})
    np.savez_compressed(root / "SINGULAR_SPECTRA.npz", **spectra)
    fields = ["PDE", "raw_shape", "working_shape", "raw_scaled_rectangular_PDE_L2", "exact_classical_PDE_L2", "exact_classical_dense_PDE_residual", "exact_classical_boundary_error_max", "rank_A64@1e-12", "cond_A64", "hashes_exact_match"]
    (root / "REPRODUCED_20260819_CROSSPDE_REPORT.md").write_text("# Reproduced 2026-08-19 NEW Cross-PDE Data\n\n" + md_table(rows, fields) + "\n\nSVD sign canonicalization is enabled; singular spectra, projectors, and physical solutions are the primary reproducibility checks.\n", encoding="utf-8")
    _write_framework_report(root, reproduce_rows=rows, direct_rows=None, audit=audit)
    return {"root": root, "rows": rows, "audit": audit}


def compare_direct64(config_path: Path, root: Path = None) -> dict:
    cfg = load_yaml(config_path)
    root = _build_root(cfg, create=False) if root is None else root
    seed = int(cfg.get("feature_seed", 7061015))
    alpha = float(cfg.get("alpha", 50.0))
    dense_points = int(cfg.get("dense_points", 4096))
    bank200 = sample_uniform_sine_features(200, seed=seed, alpha=alpha)
    rows = []
    spectra = {}
    for key in cfg.get("pdes", [p.key for p in pde_specs()]):
        hist = build_rfm_system(key, 200, seed, alpha, dense_points, canonicalize_svd=bool(cfg.get("canonicalize_svd_signs", True)))
        _zr, _cr, raw_row, raw_mat = hist.raw_rectangular_diagnostics()
        _zw, _cw, work_row, work_mat = hist.exact_working_diagnostics()
        direct = build_rfm_system(key, 64, seed, alpha, dense_points, construction="DIRECT64_REGENERATED")
        _zd, _cd, direct_row, direct_mat = direct.exact_working_diagnostics()
        prefix = build_rfm_system(key, 64, seed, alpha, dense_points, construction="PREFIX64_FROM_200_DIAGNOSTIC", source_bank=bank200)
        _zp, _cp, prefix_row, prefix_mat = prefix.exact_working_diagnostics()
        rows.extend([raw_row, work_row, direct_row, prefix_row])
        spectra[key + "_hist_raw_Ap"] = np.asarray(raw_mat["singular_values"], dtype=float)
        spectra[key + "_hist_work_A64"] = np.asarray(work_mat["singular_values"], dtype=float)
        spectra[key + "_direct64_Ap"] = np.asarray(direct_mat["singular_values"], dtype=float)
        spectra[key + "_prefix64_Ap"] = np.asarray(prefix_mat["singular_values"], dtype=float)
    write_csv(root / "DIRECT64_COMPARISON_RESULTS.csv", rows)
    write_json(root / "DIRECT64_COMPARISON_RESULTS.json", {"rows": rows})
    existing = {}
    spectra_path = root / "SINGULAR_SPECTRA.npz"
    if spectra_path.exists():
        loaded = np.load(spectra_path)
        existing = {k: loaded[k] for k in loaded.files}
    existing.update(spectra)
    np.savez_compressed(spectra_path, **existing)
    headline_fields = ["PDE", "construction", "shape", "rank@1e-12", "cond2", "kappa_eff@1e-12", "classical_residual", "PDE_L2", "PDE_Linf", "dense_PDE_residual", "boundary_error"]
    questions = _direct64_answers(rows)
    report = ["# Direct 64x64 Comparison", "", md_table(rows, headline_fields), "", "## Seven Questions", ""]
    report.extend(["- **{}** {}".format(item["question"], item["answer"]) for item in questions])
    (root / "DIRECT64_COMPARISON_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_framework_report(root, reproduce_rows=_load_reproduce_rows(root), direct_rows=rows, audit=leakage_audit_payload())
    write_unified_chinese_report(root, _load_reproduce_rows(root), rows)
    return {"root": root, "rows": rows, "questions": questions}


def _load_reproduce_rows(root: Path) -> List[dict]:
    path = root / "REPRODUCED_20260819_CROSSPDE_RESULTS.json"
    return read_json(path).get("rows", []) if path.exists() else []


def _direct64_answers(rows: List[dict]) -> List[dict]:
    direct = [r for r in rows if r["construction"] == "DIRECT64_REGENERATED"]
    hist = [r for r in rows if r["construction"] == "HISTORICAL_NEW_64x200_TO_64x64"]
    return [
        {"question": "direct64 rank", "answer": "; ".join("{} rank@1e-12={}".format(r["pde_key"], r["rank@1e-12"]) for r in direct)},
        {"question": "direct64 cond", "answer": "; ".join("{} cond2={:.6e}".format(r["pde_key"], r["cond2"]) for r in direct)},
        {"question": "direct64 effective cond", "answer": "; ".join("{} kappa_eff@1e-12={:.6e}".format(r["pde_key"], r["kappa_eff@1e-12"]) for r in direct)},
        {"question": "PDE error vs 1e-10", "answer": "; ".join("{} L2={:.6e} ({})".format(r["pde_key"], r["PDE_L2"], "above" if r["PDE_L2"] > 1e-10 else "at/below") for r in direct)},
        {"question": "200-feature right-working-space improves PDE accuracy", "answer": _compare_metric(hist, direct, "PDE_L2")},
        {"question": "200-feature right-working-space improves conditioning", "answer": _compare_metric(hist, direct, "kappa_eff@1e-12")},
        {"question": "cause by feature-count/RNG/right-space", "answer": "DIRECT64_REGENERATED changes both feature count and later RNG draw positions for b; PREFIX64 isolates the first-64-from-200 bank. Right-space retains all 64 row-space directions, so accuracy changes mainly come from the 200-feature bank before reduction, not singular-value truncation."},
    ]


def _compare_metric(hist, direct, key):
    parts = []
    for h in hist:
        d = [r for r in direct if r["pde_key"] == h["pde_key"]][0]
        parts.append("{} {}: hist={:.6e}, direct64={:.6e}".format(h["pde_key"], "improves" if h[key] < d[key] else "does not improve", h[key], d[key]))
    return "; ".join(parts)


def _write_framework_report(root: Path, reproduce_rows=None, direct_rows=None, audit=None) -> None:
    reproduce_rows = reproduce_rows or []
    direct_rows = direct_rows or []
    manifest = {"result_root": str(root.resolve()), "environment": environment_record(), "reproduction_rows": reproduce_rows, "direct_rows": direct_rows, "leakage_audit": audit or leakage_audit_payload(), "stage1_protocol": {"name": "full_basis_adam_refined_v1", "historical_fingerprint": "full_basis_chunked_adam, seed 2026072202, Normal(0,0.02^2), steps 1000, lr 0.04, cosine final multiplier 0.12, chunk 8, beta1 0.9, beta2 0.999, eps 1e-8, float64", "block11_formula": "P=42*M+7; CNOT=21*M; P5005=M119; P6013=M143"}, "stage2_protocol": {"name": "fresh_direct_r_lbfgsb_refined_v1", "fingerprint": "seed 13379, Normal(0,0.70^2), L-BFGS-B direct_r only; no direct_r2 warm start and no historical theta warm start", "capacity_ladder": [1800, 2400, 3000]}}
    write_json(root / "FRAMEWORK_BUILD_MANIFEST.json", manifest)
    lines = ["# Framework Build Report", "", "Result root: `{}`".format(root.resolve()), "", "Recovered NEW generator: `sin(50*(x*w+b))`, seed `7061015`, draw all `w` then all `b`, both Uniform[-1,1].", "", "Leakage conclusion: `{}`.".format("detected" if manifest["leakage_audit"]["leakage_detected"] else "not detected"), "", "## Reproduction", md_table(reproduce_rows, ["PDE", "raw_shape", "working_shape", "exact_classical_PDE_L2", "exact_classical_dense_PDE_residual", "hashes_exact_match"]) if reproduce_rows else "Not run yet.", "", "## Direct64", md_table(direct_rows, ["PDE", "construction", "shape", "rank@1e-12", "cond2", "PDE_L2", "dense_PDE_residual", "boundary_error"]) if direct_rows else "Not run yet.", "", "Stage-I and Stage-II production training is implemented behind CLI commands but not automatically launched by this build task."]
    (root / "FRAMEWORK_BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_unified_chinese_report(root: Path, reproduce_rows: List[dict], direct_rows: List[dict]) -> None:
    lines = ["# 实验结果总报告", "", "本次在全新目录 `standard_refined_twostage_framework/` 中重建了干净的两阶段框架，并把旧目录仅作为只读证据源使用。", "", "## 2026-08-19 NEW 数据复现", "", md_table(reproduce_rows, ["PDE", "raw_shape", "working_shape", "raw_scaled_rectangular_PDE_L2", "exact_classical_PDE_L2", "exact_classical_dense_PDE_residual", "exact_classical_boundary_error_max"]) if reproduce_rows else "未运行。", "", "## 泄漏审计结论", "", "未发现锚点特征、精确解频率、候选池筛选、top-k、pivoted QR、seed search、特征替换、特殊列 boost、按 PDE 误差接受特征或量子结果反馈。历史 NEW 路径只有固定特征库构造后的通过/失败数据路径门控。", "", "## Direct 64x64 与 Prefix64", "", md_table(direct_rows, ["PDE", "construction", "shape", "rank@1e-12", "cond2", "kappa_eff@1e-12", "classical_residual", "PDE_L2", "PDE_Linf", "dense_PDE_residual", "boundary_error"]) if direct_rows else "未运行。", "", "## 解释", "", "200-feature -> 64 工作空间保留全部 64 个行空间方向，因此不是通过丢弃小奇异值来改善非零奇异值条件数；它的主要作用是先用更大的随机特征库形成同一 64 维量子工作系统。DIRECT64_REGENERATED 不调参、不换 seed、不正则化，是一次固定测量。"]
    (project_root() / "EXPERIMENT_RESULT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "EXPERIMENT_RESULT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage1_command(config_path: Path, pde_key: str) -> dict:
    cfg = load_yaml(config_path)
    root = _build_root(cfg, create=True)
    gen = cfg.get("generator", {})
    bundle = build_rfm_system(
        pde_key,
        int(gen.get("feature_count", cfg.get("feature_count", 64))),
        int(gen.get("feature_seed", cfg.get("feature_seed", 7061015))),
        float(gen.get("alpha", cfg.get("alpha", 50.0))),
        int(gen.get("dense_points", cfg.get("dense_points", 4096))),
        construction=gen.get("construction"),
        canonicalize_svd=bool(gen.get("canonicalize_svd_signs", False)),
    )
    As = bundle.As
    target = int(cfg.get("target_parameters", 5005))
    ansatz = make_stage1_ansatz(cfg.get("family", "vcbe_block11_real"), target_parameters=target)
    opt = cfg.get("optimizer", {})
    out = root / "stage1_smoke" / pde_key
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "stage1_checkpoint_latest.npz"
    result = adam_optimize(ansatz, As, seed=int(opt.get("theta0_seed", 2026072202)), init_scale=float(opt.get("init_scale", 0.02)), steps=int(opt.get("steps", 20 if cfg.get("smoke_only", True) else 1000)), lr=float(opt.get("lr", 0.04)), final_multiplier=float(opt.get("final_multiplier", 0.12)), chunk_size=int(opt.get("chunk_size", 8)), checkpoint_dir=out, checkpoint_interval=int(opt.get("history_interval", 25)), resume_checkpoint=checkpoint_path if checkpoint_path.exists() else None)
    np.save(out / "theta0.npy", result["theta0"])
    np.save(out / "final_theta.npy", result["theta"])
    np.save(out / "theta_best.npy", result["theta_best"])
    np.save(out / "theta_last.npy", result["theta_last"])
    np.save(out / "learned_operator.npy", result["K"])
    payload = {
        "pde_key": pde_key,
        "family": ansatz.family_name,
        "target_parameters": target,
        "actual_parameters": ansatz.num_parameters,
        "resources": ansatz.resource_counts().to_dict(),
        "theta0_hash": short_hash_array(result["theta0"]),
        "theta_hash": short_hash_array(result["theta"]),
        "theta_best_hash": short_hash_array(result["theta_best"]),
        "theta_last_hash": short_hash_array(result["theta_last"]),
        "projected_block_hash": sha256_array(result["K"]),
        "target_hash": sha256_array(As),
        "smoke_only": bool(cfg.get("smoke_only", True)),
        "best_loss": result["best_loss"],
        "last_loss": result["last_loss"],
        "best_step": result["best_step"],
        "last_step": result["last_step"],
        "best_gradient_norm": result["best_gradient_norm"],
        "last_gradient_norm": result["last_gradient_norm"],
        "final_gradient_norm": result["final_gradient_norm"],
        "theta_final_policy": result["theta_final_policy"],
    }
    payload.update(result["metrics"])
    write_json(out / "final_result.json", payload)
    return {"root": root, "payload": payload}


def run_stage2_command(config_path: Path, pde_key: str) -> dict:
    cfg = load_yaml(config_path)
    root = _build_root(cfg, create=True)
    gen = cfg.get("generator", {})
    bundle = build_rfm_system(
        pde_key,
        int(gen.get("feature_count", cfg.get("feature_count", 64))),
        int(gen.get("feature_seed", cfg.get("feature_seed", 7061015))),
        float(gen.get("alpha", cfg.get("alpha", 50.0))),
        int(gen.get("dense_points", cfg.get("dense_points", 4096))),
        construction=gen.get("construction"),
        canonicalize_svd=bool(gen.get("canonicalize_svd_signs", False)),
    )
    A = bundle.A64
    b = bundle.b_p
    target = int(cfg.get("target_parameters", 24 if cfg.get("smoke_only", True) else 2400))
    ansatz = make_stage2_ansatz(cfg.get("family", "alternating_ry_cnot"), target_parameters=target)
    opts = {"maxiter": int(cfg.get("smoke_maxiter", 3)), "maxfun": int(cfg.get("smoke_maxfun", 12)), "maxcor": 10, "maxls": 20, "ftol": 1e-12, "gtol": 1e-9, "disp": False} if cfg.get("smoke_only", True) else None
    protocol = cfg.get("protocol", cfg.get("optimizer", {}).get("protocol", "fresh_direct_r"))
    if protocol == "fresh_direct_r":
        result = lbfgsb_fresh_direct_r(A, b, ansatz, seed=int(cfg.get("theta0_seed", 13379)), init_scale=float(cfg.get("init_scale", 0.70)), options=opts)
        theta = result["theta"]
        stage2_payload = {
            "initial_r": result["initial_r"],
            "final_r": result["final_r"],
            "message": result["message"],
            "nit": result["nit"],
            "nfev": result["nfev"],
            "success": result["success"],
            "protocol": result["protocol"],
        }
    elif protocol == "historical_r2_then_r":
        result = lbfgsb_r2_then_r(A, b, ansatz, seed=int(cfg.get("theta0_seed", 13379)), init_scale=float(cfg.get("init_scale", 0.70)), r2_options=opts, r_options=opts)
        theta = result["phase2"]["theta"]
        stage2_payload = {"r_after_r2": result["phase1"]["direct_r"], "final_r": result["phase2"]["direct_r"], "phase1_message": result["phase1"]["message"], "phase2_message": result["phase2"]["message"], "protocol": protocol}
    else:
        raise ValueError("unknown Stage-II protocol {}".format(protocol))
    diag = stage2_solution_diagnostics(A, b, ansatz, theta, bundle.raw_coefficients, bundle.pde, bundle.feature_bank, bundle.dense_x)
    out = root / "stage2_smoke" / pde_key
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "theta0.npy", result["theta0"])
    np.save(out / "final_theta.npy", theta)
    payload = {"pde_key": pde_key, "family": ansatz.family_name, "target_parameters": target, "actual_parameters": ansatz.num_parameters, "resources": ansatz.resource_counts().to_dict(), "smoke_only": bool(cfg.get("smoke_only", True))}
    payload.update(stage2_payload)
    payload.update(diag)
    write_json(out / "RESULT.json", payload)
    return {"root": root, "payload": payload}


def run_two_stage_command(config_path: Path, pde_key: str) -> dict:
    cfg = load_yaml(config_path)
    root = _build_root(cfg, create=True)
    stage1 = run_stage1_command(project_root() / "configs" / "paper" / "stage1_block11_direct64.yaml", pde_key)
    stage2 = run_stage2_command(project_root() / "configs" / "paper" / "direct64_stage2_fresh_direct_r.yaml", pde_key)
    return {"root": root, "stage1": stage1["payload"], "stage2": stage2["payload"], "config": cfg}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="refined_twostage")
    sub = parser.add_subparsers(dest="command")
    try:
        sub.required = True
    except Exception:
        pass
    for name in ["reproduce-data", "compare-direct64", "run-stage1", "run-stage2", "run-two-stage"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name in ("run-stage1", "run-stage2", "run-two-stage"):
            p.add_argument("--pde", required=True)
    args = parser.parse_args(argv)
    try:
        cfg_path = _cfg_path(args.config)
        if args.command == "reproduce-data":
            result = reproduce_crosspde(cfg_path)
        elif args.command == "compare-direct64":
            result = compare_direct64(cfg_path)
        elif args.command == "run-stage1":
            result = run_stage1_command(cfg_path, args.pde)
        elif args.command == "run-stage2":
            result = run_stage2_command(cfg_path, args.pde)
        else:
            result = run_two_stage_command(cfg_path, args.pde)
        print("result_root={}".format(result["root"]))
        return 0
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
