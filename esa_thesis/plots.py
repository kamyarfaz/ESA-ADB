"""Plots components of the Mission 1 research pipeline.

Extracted without changing numerical behavior; see docs/thesis-research-context.md.
"""
from __future__ import annotations

from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import (MAX_ZOOM_EVENTS, ZOOM_PAD)
from .metrics import (find_events, ov)


def ds_downsample(arr: np.ndarray, max_points: int = 60000):
    if len(arr) <= max_points:
        return np.arange(len(arr)), arr
    step = int(math.ceil(len(arr) / max_points))
    idx  = np.arange(0, len(arr), step)
    return idx, arr[idx]


def plot_overview(path: Path, score, pred, gt, thr, title):
    xs, sc = ds_downsample(score); xg, gt2 = ds_downsample(gt); _, pr2 = ds_downsample(pred)
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(title, fontsize=10)
    ax[0].plot(xs, sc, lw=0.6, label="score"); ax[0].axhline(thr, ls="--", lw=1, label="threshold"); ax[0].legend(); ax[0].grid(alpha=.25)
    ax[1].fill_between(xg, gt2.astype(float), alpha=.65, label="GT"); ax[1].fill_between(xg, -pr2.astype(float)*.6, alpha=.55, label="Prediction"); ax[1].set_ylim(-.8, 1.2); ax[1].legend(); ax[1].grid(alpha=.25)
    tp = ((pred==1)&(gt==1)).astype(np.int8); fp = ((pred==1)&(gt==0)).astype(np.int8); fn = ((pred==0)&(gt==1)).astype(np.int8)
    xt, tp2 = ds_downsample(tp); _, fp2 = ds_downsample(fp); _, fn2 = ds_downsample(fn)
    ax[2].fill_between(xt, tp2, alpha=.55, label="TP"); ax[2].fill_between(xt, -fp2, alpha=.55, label="FP"); ax[2].fill_between(xt, fn2*.5, alpha=.55, label="FN"); ax[2].set_ylim(-1.2, 1.2); ax[2].legend(); ax[2].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_channels(path: Path, per_ch, features, gt, thr, title):
    n   = min(per_ch.shape[0], 16)
    top = np.argsort(per_ch.max(axis=1))[::-1][:n]
    fig, axes = plt.subplots(n, 1, figsize=(18, max(6, 3*n)), sharex=True)
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=10); xg, gt2 = ds_downsample(gt)
    for r, ci in enumerate(top):
        xs, sc = ds_downsample(per_ch[ci]); ax = axes[r]
        ax.plot(xs, sc, lw=.5, label=f"{features[ci]} score"); ax.axhline(thr, ls="--", lw=.8, alpha=.5)
        ymax = float(np.nanmax(sc)) if len(sc) else 1.0
        ax.fill_between(xg, gt2.astype(float)*ymax, alpha=.18, label="GT")
        ax.set_ylabel(features[ci], fontsize=8); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_zooms(plot_dir: Path, score, pred, gt, thr, prefix):
    evs  = find_events(gt); pevs = find_events(pred)
    rows = sorted(
        [(i, ev, any(ov(ev, p) for p in pevs)) for i, ev in enumerate(evs)],
        key=lambda x: (x[2], x[0])
    )[:MAX_ZOOM_EVENTS]
    for i, (s, e), det in rows:
        l, r = max(0, s-ZOOM_PAD), min(len(gt), e+ZOOM_PAD); x = np.arange(l, r)
        fig, ax = plt.subplots(3, 1, figsize=(18, 9), sharex=True)
        status = "detected" if det else "MISSED"
        fig.suptitle(f"{prefix} event {i:03d} [{s},{e}] {status}", fontsize=10)
        ax[0].plot(x, score[l:r], lw=.7); ax[0].axhline(thr, ls="--", lw=1); ax[0].axvspan(s, e, alpha=.15); ax[0].grid(alpha=.25)
        ax[1].fill_between(x, gt[l:r].astype(float), alpha=.65, label="GT"); ax[1].fill_between(x, -pred[l:r].astype(float)*.6, alpha=.55, label="Pred"); ax[1].set_ylim(-.8, 1.2); ax[1].legend(); ax[1].grid(alpha=.25)
        tp = ((pred[l:r]==1)&(gt[l:r]==1)).astype(np.int8); fp = ((pred[l:r]==1)&(gt[l:r]==0)).astype(np.int8); fn = ((pred[l:r]==0)&(gt[l:r]==1)).astype(np.int8)
        ax[2].fill_between(x, tp, alpha=.55, label="TP"); ax[2].fill_between(x, -fp, alpha=.55, label="FP"); ax[2].fill_between(x, fn*.5, alpha=.55, label="FN"); ax[2].set_ylim(-1.2, 1.2); ax[2].legend(); ax[2].grid(alpha=.25)
        fig.tight_layout(); fig.savefig(plot_dir / f"event_zoom_{i:03d}_{status.lower()}.jpg", dpi=120, bbox_inches="tight"); plt.close(fig)
