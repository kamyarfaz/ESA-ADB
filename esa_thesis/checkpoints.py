"""Checkpoints components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import pandas as pd
import time
import torch
from .config import (TRAIN_MLP)


def save_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_training_history(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except Exception:
        return []


def save_training_history(path: Path, hist: List[Dict]) -> None:
    pd.DataFrame(hist).to_csv(path, index=False)


def save_training_checkpoint(path, model, opt, sched, epoch, best_val, best_epoch, hist, features):
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "epoch":      int(epoch),
        "best_val":   float(best_val),
        "best_epoch": int(best_epoch),
        "history":    hist,
        "features":   features,
        "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def save_best_checkpoint(path, model, epoch, best_val, row, features):
    torch.save({
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "epoch":    int(epoch),
        "best_val": float(best_val),
        "row":      row,
        "features": features,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def completed_run(run_dir: Path, plot_dir: Path) -> bool:
    # AE must always be complete
    ae_done = (
        (run_dir  / "summary.json").exists()
        and (run_dir  / "selected_thresholds.csv").exists()
        and (run_dir  / "threshold_sweep_val_to_test.csv").exists()
        and (run_dir  / "test_score.npy").exists()
        and (plot_dir / "overview.jpg").exists()
    )
    if not ae_done:
        return False
    # If MLP is enabled, also check MLP scores exist
    if TRAIN_MLP:
        mlp_done = (
            (run_dir / "mlp_test_score.npy").exists()
            and (run_dir / "ens_test_score.npy").exists()
        )
        return mlp_done
    return True
