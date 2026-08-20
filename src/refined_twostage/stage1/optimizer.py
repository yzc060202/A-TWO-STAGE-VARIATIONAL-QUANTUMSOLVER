"""Full-basis Adam optimizer for Stage I."""

from __future__ import annotations

import math
import time
from pathlib import Path
import numpy as np

from refined_twostage.io.artifacts import read_json, write_json
from refined_twostage.io.hashing import short_hash_array
from refined_twostage.stage1.evaluator import stage1_metrics


def cosine_lr(step, steps, lr, final_multiplier):
    frac = (int(step) - 1) / max(1, int(steps) - 1)
    return float(lr) * (float(final_multiplier) + (1.0 - float(final_multiplier)) * 0.5 * (1.0 + math.cos(math.pi * frac)))


def _write_checkpoint(checkpoint_dir, payload):
    if checkpoint_dir is None:
        return
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "stage1_checkpoint_latest.npz",
        theta=payload["theta"],
        theta_pre_update=payload["theta_pre_update"],
        theta0=payload["theta0"],
        m=payload["m"],
        v=payload["v"],
        best_theta=payload["best_theta"],
        best_grad=payload["best_grad"],
        step=np.asarray(payload["step"], dtype=np.int64),
        steps=np.asarray(payload["steps"], dtype=np.int64),
        best_loss=np.asarray(payload["best_loss"], dtype=float),
        best_step=np.asarray(payload["best_step"], dtype=np.int64),
    )
    np.save(out / "latest_theta_checkpoint.npy", payload["theta"])
    np.save(out / "latest_pre_update_theta_checkpoint.npy", payload["theta_pre_update"])
    np.save(out / "best_theta_checkpoint.npy", payload["best_theta"])
    write_json(
        out / "stage1_checkpoint_status.json",
        {
            "step": int(payload["step"]),
            "steps": int(payload["steps"]),
            "loss": float(payload["loss"]),
            "gradient_norm": float(payload["gradient_norm"]),
            "learning_rate": float(payload["learning_rate"]),
            "best_loss": float(payload["best_loss"]),
            "best_step": int(payload["best_step"]),
            "theta_hash": short_hash_array(payload["theta"]),
            "theta_pre_update_hash": short_hash_array(payload["theta_pre_update"]),
            "best_theta_hash": short_hash_array(payload["best_theta"]),
            "elapsed_seconds": float(payload["elapsed_seconds"]),
            "state": "running",
        },
    )


def _load_checkpoint(checkpoint_path: Path, ansatz) -> dict:
    data = np.load(checkpoint_path)
    theta = np.asarray(data["theta"], dtype=float)
    if theta.size != int(ansatz.num_parameters):
        raise ValueError("checkpoint theta has {} parameters; ansatz expects {}".format(theta.size, ansatz.num_parameters))
    status_path = checkpoint_path.with_name("stage1_checkpoint_status.json")
    status = read_json(status_path) if status_path.exists() else {}
    return {
        "theta": theta,
        "theta0": np.asarray(data["theta0"], dtype=float),
        "m": np.asarray(data["m"], dtype=float),
        "v": np.asarray(data["v"], dtype=float),
        "best_theta": np.asarray(data["best_theta"], dtype=float),
        "best_grad": np.asarray(data["best_grad"], dtype=float),
        "step": int(data["step"]) if "step" in data.files else int(status.get("step", 0)),
        "best_loss": float(data["best_loss"]) if "best_loss" in data.files else float(status.get("best_loss", "inf")),
        "best_step": int(data["best_step"]) if "best_step" in data.files else int(status.get("best_step", 0)),
    }


def adam_optimize(ansatz, target, seed=2026072202, init_scale=0.02, steps=1000, lr=0.04, final_multiplier=0.12, beta1=0.9, beta2=0.999, eps=1e-8, chunk_size=8, history_interval=25, checkpoint_dir=None, checkpoint_interval=None, resume_checkpoint=None):
    if resume_checkpoint is not None:
        state = _load_checkpoint(Path(resume_checkpoint), ansatz)
        theta = state["theta"].copy()
        theta0 = state["theta0"].copy()
        m = state["m"].copy()
        v = state["v"].copy()
        best_loss = float(state["best_loss"])
        best_theta = state["best_theta"].copy()
        best_step = int(state["best_step"])
        best_grad = state["best_grad"].copy()
        start_step = int(state["step"]) + 1
    else:
        theta = np.random.default_rng(int(seed)).normal(scale=float(init_scale), size=int(ansatz.num_parameters)).astype(float)
        theta0 = theta.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        best_loss = float("inf")
        best_theta = theta.copy()
        best_step = 0
        best_grad = np.zeros_like(theta)
        start_step = 1
    history = []
    t0 = time.time()
    for step in range(start_step, int(steps) + 1):
        theta_pre_update = theta.copy()
        loss, grad = ansatz.loss_and_grad_full_basis(theta, target, chunk_size=chunk_size, evaluator_mode="full_batch_tape")
        if loss < best_loss:
            best_loss = float(loss)
            best_theta = theta_pre_update.copy()
            best_step = int(step)
            best_grad = grad.copy()
        step_lr = cosine_lr(step, steps, lr, final_multiplier)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        theta = theta - step_lr * (m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)
        grad_norm = float(np.linalg.norm(grad))
        elapsed = float(time.time() - t0)
        if step == 1 or step == int(steps) or step % int(history_interval) == 0:
            history.append({
                "step": int(step),
                "loss": float(loss),
                "e_F": float(math.sqrt(max(float(loss), 0.0))),
                "gradient_norm": grad_norm,
                "learning_rate": float(step_lr),
                "best_loss": float(best_loss),
                "theta_hash": short_hash_array(theta),
                "pre_update_theta_hash": short_hash_array(theta_pre_update),
                "post_update_theta_hash": short_hash_array(theta),
                "elapsed_seconds": elapsed,
            })
        interval = int(checkpoint_interval or history_interval)
        if checkpoint_dir is not None and (step == 1 or step == int(steps) or step % interval == 0):
            _write_checkpoint(
                checkpoint_dir,
                {
                    "step": int(step),
                    "steps": int(steps),
                    "loss": float(loss),
                    "gradient_norm": grad_norm,
                    "learning_rate": float(step_lr),
                    "best_loss": float(best_loss),
                    "best_step": int(best_step),
                    "theta": theta,
                    "theta_pre_update": theta_pre_update,
                    "theta0": theta0,
                    "m": m,
                    "v": v,
                    "best_theta": best_theta,
                    "best_grad": best_grad,
                    "elapsed_seconds": elapsed,
                },
            )
    last_theta = theta.copy()
    last_loss, last_grad = ansatz.loss_and_grad_full_basis(last_theta, target, chunk_size=chunk_size, evaluator_mode="full_batch_tape")
    selected_theta = best_theta
    K = ansatz.extract_projected_block(selected_theta)
    if checkpoint_dir is not None:
        write_json(
            Path(checkpoint_dir) / "stage1_checkpoint_status.json",
            {
                "step": int(steps),
                "steps": int(steps),
                "best_loss": float(best_loss),
                "best_step": int(best_step),
                "last_loss": float(last_loss),
                "last_gradient_norm": float(np.linalg.norm(last_grad)),
                "best_gradient_norm": float(np.linalg.norm(best_grad)),
                "theta_hash": short_hash_array(selected_theta),
                "theta_final_policy": "best_theta",
                "state": "complete",
            },
        )
    return {
        "theta0": theta0,
        "theta": selected_theta,
        "theta_best": best_theta,
        "theta_last": last_theta,
        "K": K,
        "history": history,
        "final_loss": float(best_loss),
        "best_loss": float(best_loss),
        "last_loss": float(last_loss),
        "best_step": int(best_step),
        "last_step": int(steps),
        "final_gradient_norm": float(np.linalg.norm(best_grad)),
        "best_gradient_norm": float(np.linalg.norm(best_grad)),
        "last_gradient_norm": float(np.linalg.norm(last_grad)),
        "theta_final_policy": "best_theta",
        "metrics": stage1_metrics(K, target),
    }
