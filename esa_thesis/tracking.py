"""
wandb_utils.py -- lightweight, robust Weights & Biases logging for the
ESA-ADB Mission 1 reconstruction-AE sweep.

Design goals (all deliberate for this project):
  * Never crash a training run. If wandb is missing, offline, or misconfigured,
    every call degrades to a plain stdout print. A logging library must not be
    able to kill a 40-epoch GPU job.
  * Work whether or not ReconfigurableServer has outbound internet. The default
    mode is "offline": everything is written under ./wandb/offline-run-* and can
    be pushed later with `wandb sync wandb/offline-run-*`. If the box does have
    internet, set the env var WANDB_MODE=online to stream live -- no code change.
  * Make channel reduction first-class. The channel list, its size, and the
    selection method are logged as run config, so in the UI you can group by
    `channel_selection` and sort/filter by `n_channels`. That is exactly the
    "organized, defensible" view the supervisor wants.

Python 3.9 compatible (uses typing.Optional, not X | None).

Typical use inside the sweep, once per run:

    from esa_thesis.tracking import WandbRun

    wb = WandbRun(
        run_name=spec.name,                 # e.g. "clust_med_t03_6ch"
        config=hparams_dict,                # SEQ_LEN, D_MODEL, EPOCHS, LR, ...
        channels=spec.channels,             # e.g. [3, 5, 13, 29, 48, 57]
        selection=spec.selection,           # e.g. "clust_med_t03" (how chosen)
        group="cluster_rep",                # groups the leak-free runs together
        project="esa-adb-mission1",
    )
    for epoch in range(EPOCHS):
        train_loss = ...                    # your existing loop
        val_loss   = ...                    # or None if not computed this epoch
        wb.log_epoch(epoch, train_loss=train_loss, val_loss=val_loss)

    # after leak-free val-only threshold selection + rescore:
    wb.log_summary({
        "val_esa_f05": val_f05,
        "val_pred_rate": val_rate,
        "A_F0.5": a_f05, "B_F0.5": b_f05,
        "A_FPe": a_fpe, "n_channels": len(spec.channels),
    })
    wb.finish()
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

# Default to offline BEFORE importing wandb, so a server with no internet still
# records everything to disk. Override at launch with:  WANDB_MODE=online python ...
os.environ.setdefault("WANDB_MODE", "offline")

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except Exception:  # ImportError, or a broken install -- either way, no-op.
    _WANDB_AVAILABLE = False


class WandbRun:
    """One wandb run with safe fallbacks. All methods are no-throw."""

    def __init__(
        self,
        run_name: str,
        config: Mapping[str, Any],
        channels: Sequence[int],
        selection: str = "unknown",
        group: Optional[str] = None,
        project: str = "esa-adb-mission1",
        entity: Optional[str] = None,
    ) -> None:
        self._run = None
        self._name = run_name

        if not _WANDB_AVAILABLE:
            print(f"[wandb_utils] wandb unavailable; '{run_name}' -> stdout only.")
            return

        full_config = dict(config)
        full_config.update(
            {
                "channels": list(channels),
                "n_channels": len(channels),
                "channel_selection": selection,
            }
        )
        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                group=group,
                config=full_config,
                reinit=True,
            )
            # Make "epoch" the x-axis for the loss curves in the UI.
            self._run.define_metric("epoch")
            self._run.define_metric("train_loss", step_metric="epoch")
            self._run.define_metric("val_loss", step_metric="epoch")
        except Exception as e:  # offline dir issues, auth, etc.
            print(f"[wandb_utils] wandb.init failed ({e}); continuing without wandb.")
            self._run = None

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        **extra: float,
    ) -> None:
        payload = {"epoch": int(epoch), "train_loss": float(train_loss)}
        if val_loss is not None:
            payload["val_loss"] = float(val_loss)
        for k, v in extra.items():
            payload[k] = float(v)

        if self._run is not None:
            self._run.log(payload)
        else:
            body = " ".join(f"{k}={v:.5f}" for k, v in payload.items() if k != "epoch")
            print(f"[{self._name}] epoch {epoch}: {body}")

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        """Final numbers of record for the run (val + A/B protocol F0.5, etc.)."""
        if self._run is not None:
            for k, v in metrics.items():
                self._run.summary[k] = v
        else:
            body = " ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"[{self._name}] summary: {body}")

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def sync_hint() -> str:
    """Printable reminder for uploading offline runs once internet is available."""
    return (
        "Offline runs are under ./wandb/. To upload later:\n"
        "    wandb login            # once, if not already\n"
        "    wandb sync wandb/offline-run-*"
    )
