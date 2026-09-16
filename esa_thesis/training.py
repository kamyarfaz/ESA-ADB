"""Training components of the Mission 1 research pipeline.

Uses corrected annotation-ID/duration evaluation; see docs/research/corrected-evaluation.md.
"""
from __future__ import annotations

from .tracking import WandbRun
from .protocol import validation_split, evaluator_for, read_timestamps, guard_run, manifest
from .evaluation.esa import timestamps_ns, METRIC_VERSION
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple
import gc
import json
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from .augmentation import (make_pseudo_anomaly)
from .checkpoints import (completed_run, load_json, load_training_history, save_best_checkpoint, save_json, save_training_checkpoint, save_training_history)
from .config import (AMP, BATCH_SIZE, DEVICE, DROPOUT, D_MODEL, EPOCHS, GRAD_CLIP, LR, MAX_TRAIN_WINDOWS, MERGE_GAPS, MIN_DURS, MLP_BATCH, MLP_EPOCHS, MLP_HIDDEN, MLP_LAYERS, MLP_LR, N_HEADS, N_LAYERS, OUT_ROOT, PATCH_SIZE, PA_LAMBDA, PA_MARGIN, RUN_SPECS, SCORE_STRIDE, SEED, SEQ_LEN, STRIDE, TEST_FILE, TRAIN_FILE, TRAIN_MLP, VAL_MONTHS, WANDB_PROJECT, WEIGHT_DECAY)
from .data import (ForecastWindows, NormalWindows, RobustChannelScaler, get_y_any, load_frame, read_columns)
from .metrics import (find_events, metrics)
from .models import (MLPForecaster, MultivariateAE)
from .plots import (plot_channels, plot_overview, plot_zooms)
from .runtime import (free_memory, log, set_seed)
from .scoring import (ensemble_scores, score_series, score_series_mlp)
from .thresholds import (pick_leakfree_row, postprocess, select_rows, sweep_thresholds, thresholds)


def _channels_from_features(features: List[str]) -> List[int]:
    """Recover integer channel ids from 'channel_N' feature names for logging."""
    out = []
    for f in features:
        try:
            out.append(int(str(f).split("_")[-1]))
        except (ValueError, IndexError):
            pass
    return out


def train_one(spec: Dict, out_base: Path, available_cols: List[str]) -> Dict:
    set_seed(SEED)

    run      = spec["run"]
    features = [f for f in spec["features"] if f in available_cols]
    if not features:
        raise RuntimeError(f"No features for {run}")
    use_pa   = bool(spec.get("pseudo", False))

    run_dir  = out_base / run
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_base / "plots" / run
    plot_dir.mkdir(parents=True, exist_ok=True)

    guard_run(run_dir, manifest(spec))

    if completed_run(run_dir, plot_dir):
        log(f"SKIP completed run {run}")
        return load_json(run_dir / "summary.json")

    log(f"Run {run}: {features}")
    if use_pa:
        log(f"  [pseudo-anomaly training ENABLED]  lambda={PA_LAMBDA} margin={PA_MARGIN}")

    wb = WandbRun(
        run_name=run,
        config=dict(
            seq_len=SEQ_LEN, patch_size=PATCH_SIZE, d_model=D_MODEL,
            n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
            score_stride=SCORE_STRIDE, max_train_windows=MAX_TRAIN_WINDOWS,
            train_mlp=TRAIN_MLP, mlp_epochs=MLP_EPOCHS,
        ),
        channels=_channels_from_features(features),
        selection=spec.get("selection", spec.get("group", "unknown")),
        group=spec.get("group"),
        project=WANDB_PROJECT,
    )

    df      = load_frame(TRAIN_FILE, features)
    y_full  = get_y_any(df)
    x_full  = df[features].to_numpy(np.float32)
    train_times = timestamps_ns(df["timestamp"])
    del df

    split = validation_split(train_times)
    val_evaluator = evaluator_for(train_times[split:])
    test_evaluator = evaluator_for(read_timestamps(TEST_FILE))
    xtr_raw, ytr = x_full[:split], y_full[:split]
    xv_raw,  yv  = x_full[split:], y_full[split:]

    scaler = RobustChannelScaler().fit(xtr_raw, ytr)
    xtr    = scaler.transform(xtr_raw)
    xv     = scaler.transform(xv_raw)

    del x_full, y_full, xtr_raw, xv_raw
    gc.collect()

    log(f"rows train={len(xtr):,} val={len(xv):,} val_events={len(find_events(yv))}")

    ds_norm = NormalWindows(xtr, ytr)
    dl = DataLoader(ds_norm, BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=torch.cuda.is_available())
    log(f"normal windows={len(ds_norm):,}")

    model      = MultivariateAE(len(features)).to(DEVICE)
    opt        = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit       = nn.MSELoss()
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(AMP and torch.cuda.is_available()))

    checkpoint_last = run_dir / "checkpoint_last.pt"
    checkpoint_best = run_dir / "checkpoint_best.pt"
    history_path    = run_dir / "training_history.csv"
    state_path      = run_dir / "resume_state.json"

    hist        = load_training_history(history_path)
    best_val    = -1.0
    best_epoch  = -1
    start_epoch = 1

    if checkpoint_last.exists():
        # FIX: always load to CPU first to avoid CUDA context issues after a
        # previous crash; move the model to DEVICE afterwards.
        ckpt = torch.load(checkpoint_last, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        try:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            log(f"Could not restore optimizer/scheduler; continuing with model weights only. {e}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val    = float(ckpt.get("best_val", -1.0))
        best_epoch  = int(ckpt.get("best_epoch", -1))
        hist        = ckpt.get("history", hist)
        log(f"RESUME {run}: start_epoch={start_epoch} best_epoch={best_epoch} best_val={best_val:.4f}")

    if checkpoint_best.exists():
        bckpt      = torch.load(checkpoint_best, map_location="cpu")
        best_val   = max(best_val, float(bckpt.get("best_val", -1.0)))
        best_epoch = int(bckpt.get("epoch", best_epoch))

    val_every = max(5, EPOCHS // 8)

    # Reduced quick-val grid — still covers the relevant space but 4×3=12 vs 4×3=12
    # (same counts as original; kept identical to preserve accuracy)
    QUICK_MERGE_GAPS = [0, 64, 256, 1024]
    QUICK_MIN_DURS   = [1, 16, 64]

    if start_epoch <= EPOCHS:
        for ep in range(start_epoch, EPOCHS + 1):
            t0 = time.time()
            model.train()
            losses = []

            for xb in dl:
                xb = xb.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if use_pa:
                    # Pseudo-anomaly training: reconstruct normal well, fake anomalies badly.
                    x_pa = make_pseudo_anomaly(xb)
                    with torch.cuda.amp.autocast(enabled=(AMP and torch.cuda.is_available())):
                        recon    = model(xb)
                        recon_pa = model(x_pa)
                    # loss computed in fp32 for stability at ~1e-6 error scales
                    err_norm = ((recon.float()    - xb.float())   ** 2).mean(dim=(1, 2, 3))   # (B,)
                    err_pa   = ((recon_pa.float() - x_pa.float()) ** 2).mean(dim=(1, 2, 3))   # (B,)
                    denom    = err_norm.mean().detach() + 1e-8
                    # separation hinge: pseudo error should reach >= PA_MARGIN x normal error
                    loss = err_norm.mean() + PA_LAMBDA * torch.relu(PA_MARGIN - err_pa / denom).mean()
                else:
                    with torch.cuda.amp.autocast(enabled=(AMP and torch.cuda.is_available())):
                        loss = crit(model(xb), xb)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler_amp.step(opt)
                scaler_amp.update()
                losses.append(float(loss.detach().cpu()))

            sched.step()
            row = {"epoch": ep, "train_loss": float(np.mean(losses)),
                   "seconds": float(time.time() - t0)}

            if ep % val_every == 0 or ep == EPOCHS:
                _, vs = score_series(model, xv)
                quick = []
                for thr in thresholds(vs):
                    for mg in QUICK_MERGE_GAPS:
                        for md in QUICK_MIN_DURS:
                            m = metrics(yv, postprocess((vs > thr).astype(np.int8), mg, md), evaluator=val_evaluator)
                            if m["pred_anomaly_rate"] <= 0.01 and not m["saturated"]:
                                quick.append((m["esa_f05"], m["event_f05"], m["point_f05"],
                                              thr, mg, md, m["pred_anomaly_rate"]))
                if quick:
                    quick.sort(reverse=True, key=lambda z: (z[0], z[1], z[2]))
                    q = quick[0]
                    row.update({
                        "val_esa_f05_quick":     q[0],
                        "val_event_f05_quick":   q[1],
                        "val_point_f05_quick":   q[2],
                        "val_threshold_quick":   q[3],
                        "val_merge_gap_quick":   q[4],
                        "val_min_dur_quick":     q[5],
                        "val_pred_rate_quick":   q[6],
                    })
                    if q[0] > best_val:
                        best_val, best_epoch = float(q[0]), ep
                        save_best_checkpoint(checkpoint_best, model, ep, best_val, row, features)
                        log(f"saved best checkpoint: epoch={ep} val_esa={best_val:.4f}")

                del vs
                gc.collect()

            hist.append(row)
            save_training_history(history_path, hist)
            save_training_checkpoint(checkpoint_last, model, opt, sched, ep, best_val, best_epoch, hist, features)
            save_json(state_path, {
                "run": run, "stage": "training",
                "last_epoch": int(ep), "best_epoch": int(best_epoch),
                "best_val": float(best_val),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            log(f"{run} ep={ep}/{EPOCHS} loss={row['train_loss']:.6f} "
                f"best_val_esa={best_val:.4f} sec={row['seconds']:.1f}")

            _extra = {}
            if "val_esa_f05_quick" in row:
                _extra["val_esa_f05"]   = row["val_esa_f05_quick"]
                _extra["val_pred_rate"] = row.get("val_pred_rate_quick", 0.0)
            wb.log_epoch(ep, train_loss=row["train_loss"], **_extra)

    if checkpoint_best.exists():
        # FIX: load to CPU first, then move to DEVICE
        bckpt      = torch.load(checkpoint_best, map_location="cpu")
        model.load_state_dict(bckpt["model_state_dict"])
        model.to(DEVICE)
        best_epoch = int(bckpt.get("epoch", best_epoch))
        best_val   = float(bckpt.get("best_val", best_val))
        log(f"loaded best checkpoint epoch={best_epoch} val_esa={best_val:.4f}")
    else:
        log("WARNING: no best checkpoint found; using latest model weights")

    val_score_path  = run_dir / "val_score.npy"
    val_per_ch_path = run_dir / "val_score_per_channel.npy"
    val_y_path      = run_dir / "val_y_true.npy"
    val_pred_path   = run_dir / "val_pred_mask.npy"

    if val_score_path.exists() and val_per_ch_path.exists() and val_y_path.exists():
        log(f"RESUME {run}: loading cached validation scores")
        vs  = np.load(val_score_path)
        vpc = np.load(val_per_ch_path)
        yv  = np.load(val_y_path)
    else:
        log(f"{run}: scoring validation")
        vpc, vs = score_series(model, xv)
        np.save(val_score_path,  vs.astype(np.float32))
        np.save(val_per_ch_path, vpc.astype(np.float32))
        np.save(val_y_path,      yv.astype(np.int8))

    save_json(state_path, {"run": run, "stage": "validation_scored",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    test_score_path  = run_dir / "test_score.npy"
    test_per_ch_path = run_dir / "test_score_per_channel.npy"
    test_y_path      = run_dir / "test_y_true.npy"
    test_pred_path   = run_dir / "test_pred_mask.npy"

    if test_score_path.exists() and test_per_ch_path.exists() and test_y_path.exists():
        log(f"RESUME {run}: loading cached test scores")
        ts  = np.load(test_score_path)
        tpc = np.load(test_per_ch_path)
        yt  = np.load(test_y_path)
    else:
        # FIX: force-free GPU memory before the (potentially much larger) test
        # inference pass to avoid CUDA driver segfaults caused by fragmentation.
        log(f"{run}: clearing GPU memory before test scoring")
        free_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # Reload best model weights cleanly onto the device
        if checkpoint_best.exists():
            _bckpt = torch.load(checkpoint_best, map_location="cpu")
            model.load_state_dict(_bckpt["model_state_dict"])
            del _bckpt
        model.to(DEVICE)
        model.eval()
        log(f"{run}: loading and scoring test")
        dfte    = load_frame(TEST_FILE, features)
        yt      = get_y_any(dfte)
        xt_raw  = dfte[features].to_numpy(np.float32)
        del dfte
        xt = scaler.transform(xt_raw)
        del xt_raw
        gc.collect()
        tpc, ts = score_series(model, xt)
        np.save(test_score_path,  ts.astype(np.float32))
        np.save(test_per_ch_path, tpc.astype(np.float32))
        np.save(test_y_path,      yt.astype(np.int8))

    save_json(state_path, {"run": run, "stage": "test_scored",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    # ── MLP Forecaster training + scoring ────────────────────────────────────
    mlp_val_score_path  = run_dir / "mlp_val_score.npy"
    mlp_test_score_path = run_dir / "mlp_test_score.npy"
    ens_val_score_path  = run_dir / "ens_val_score.npy"
    ens_test_score_path = run_dir / "ens_test_score.npy"

    if TRAIN_MLP:
        mlp_ckpt_last = run_dir / "mlp_checkpoint_last.pt"
        mlp_ckpt_best = run_dir / "mlp_checkpoint_best.pt"

        mlp_model = MLPForecaster(len(features)).to(DEVICE)
        mlp_opt   = torch.optim.AdamW(mlp_model.parameters(), lr=MLP_LR,
                                       weight_decay=WEIGHT_DECAY)
        mlp_crit  = nn.MSELoss()
        mlp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            mlp_opt, T_max=MLP_EPOCHS, eta_min=1e-6)

        mlp_start = 1
        mlp_best_val  = -1.0
        mlp_best_epoch = -1

        if mlp_ckpt_last.exists():
            mc = torch.load(mlp_ckpt_last, map_location="cpu")
            mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE)
            try:
                mlp_opt.load_state_dict(mc["optimizer_state_dict"])
                mlp_sched.load_state_dict(mc["scheduler_state_dict"])
            except Exception:
                pass
            mlp_start      = int(mc.get("epoch", 0)) + 1
            mlp_best_val   = float(mc.get("best_val", -1.0))
            mlp_best_epoch = int(mc.get("best_epoch", -1))
            log(f"RESUME MLP {run}: start_epoch={mlp_start} best_val={mlp_best_val:.4f}")

        # Build forecast windows from normal training data
        ds_fc  = ForecastWindows(xtr, stride=STRIDE, normal_only=True, y=ytr)
        dl_mlp = DataLoader(ds_fc, MLP_BATCH, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())
        log(f"{run}: MLP forecast windows={len(ds_fc):,}")

        mlp_val_every = max(5, MLP_EPOCHS // 6)

        if mlp_start <= MLP_EPOCHS:
            for ep in range(mlp_start, MLP_EPOCHS + 1):
                t0 = time.time()
                mlp_model.train()
                losses = []
                for ctx_b, tgt_b in dl_mlp:
                    ctx_b = ctx_b.to(DEVICE, non_blocking=True)
                    tgt_b = tgt_b.to(DEVICE, non_blocking=True)
                    mlp_opt.zero_grad(set_to_none=True)
                    pred  = mlp_model(ctx_b)
                    loss  = mlp_crit(pred, tgt_b)
                    if not torch.isfinite(loss):
                        raise RuntimeError("MLP non-finite loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(mlp_model.parameters(), GRAD_CLIP)
                    mlp_opt.step()
                    losses.append(float(loss.detach().cpu()))
                mlp_sched.step()

                if ep % mlp_val_every == 0 or ep == MLP_EPOCHS:
                    _, mlp_vs = score_series_mlp(mlp_model, xv)
                    quick_mlp = []
                    for thr in thresholds(mlp_vs):
                        for mg in [0, 64, 256]:
                            for md in [1, 16, 64]:
                                m = metrics(yv, postprocess(
                                    (mlp_vs > thr).astype(np.int8), mg, md), evaluator=val_evaluator)
                                if m["pred_anomaly_rate"] <= 0.01 and not m["saturated"]:
                                    quick_mlp.append((m["esa_f05"], thr))
                    if quick_mlp:
                        best_q = max(quick_mlp, key=lambda z: z[0])
                        if best_q[0] > mlp_best_val:
                            mlp_best_val   = best_q[0]
                            mlp_best_epoch = ep
                            torch.save({
                                "model_state_dict": {
                                    k: v.detach().cpu().clone()
                                    for k, v in mlp_model.state_dict().items()},
                                "epoch":     ep,
                                "best_val":  mlp_best_val,
                                "best_epoch": mlp_best_epoch,
                            }, mlp_ckpt_best)
                            log(f"MLP saved best: epoch={ep} val_esa={mlp_best_val:.4f}")
                    del mlp_vs; gc.collect()

                torch.save({
                    "model_state_dict":     mlp_model.state_dict(),
                    "optimizer_state_dict": mlp_opt.state_dict(),
                    "scheduler_state_dict": mlp_sched.state_dict(),
                    "epoch":      ep, "best_val":  mlp_best_val,
                    "best_epoch": mlp_best_epoch,
                }, mlp_ckpt_last)
                log(f"{run} MLP ep={ep}/{MLP_EPOCHS} loss={np.mean(losses):.6f} "
                    f"best_val_esa={mlp_best_val:.4f} sec={time.time()-t0:.1f}")

        # Load best MLP checkpoint
        if mlp_ckpt_best.exists():
            mc = torch.load(mlp_ckpt_best, map_location="cpu")
            mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE)
            log(f"loaded best MLP checkpoint epoch={mc['epoch']} val_esa={mc['best_val']:.4f}")

        # MLP val scoring
        if mlp_val_score_path.exists():
            log(f"RESUME {run}: loading cached MLP val scores")
            mlp_vs = np.load(mlp_val_score_path)
        else:
            log(f"{run}: MLP scoring validation")
            free_memory()
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            mlp_vpc, mlp_vs = score_series_mlp(mlp_model, xv)
            np.save(mlp_val_score_path,              mlp_vs.astype(np.float32))
            np.save(run_dir / "mlp_val_score_per_channel.npy",
                    mlp_vpc.astype(np.float32))

        # MLP test scoring — reload test data if needed
        if mlp_test_score_path.exists():
            log(f"RESUME {run}: loading cached MLP test scores")
            mlp_ts = np.load(mlp_test_score_path)
        else:
            log(f"{run}: MLP clearing GPU + loading test for scoring")
            free_memory()
            if torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            if mlp_ckpt_best.exists():
                mc = torch.load(mlp_ckpt_best, map_location="cpu")
                mlp_model.load_state_dict(mc["model_state_dict"])
            mlp_model.to(DEVICE); mlp_model.eval()
            # Reload test data (may have been freed after AE scoring)
            dfte_mlp   = load_frame(TEST_FILE, features)
            yt_mlp     = get_y_any(dfte_mlp)
            xt_raw_mlp = dfte_mlp[features].to_numpy(np.float32)
            del dfte_mlp
            xt_mlp = scaler.transform(xt_raw_mlp)
            del xt_raw_mlp; gc.collect()
            mlp_tpc, mlp_ts = score_series_mlp(mlp_model, xt_mlp)
            np.save(mlp_test_score_path,             mlp_ts.astype(np.float32))
            np.save(run_dir / "mlp_test_score_per_channel.npy",
                    mlp_tpc.astype(np.float32))
            del xt_mlp, mlp_tpc; gc.collect()

        # Ensemble scores (OR-logic: max of normalised AE and MLP)
        if not ens_val_score_path.exists():
            log(f"{run}: computing ensemble val score")
            ens_vs = ensemble_scores(vs, mlp_vs)
            np.save(ens_val_score_path, ens_vs.astype(np.float32))

        if not ens_test_score_path.exists():
            log(f"{run}: computing ensemble test score")
            _ae_ts  = np.load(test_score_path)
            _mlp_ts = np.load(mlp_test_score_path)
            ens_ts  = ensemble_scores(_ae_ts, _mlp_ts)
            np.save(ens_test_score_path, ens_ts.astype(np.float32))
            del _ae_ts, _mlp_ts, ens_ts; gc.collect()

        free_memory(mlp_model, mlp_opt, mlp_sched, ds_fc, dl_mlp)
        log(f"{run}: MLP + ensemble done")

    # ── end MLP block ────────────────────────────────────────────────────────

    threshold_path = run_dir / "threshold_sweep_val_to_test.csv"
    if threshold_path.exists():
        log(f"RESUME {run}: loading cached threshold sweep")
        thdf = pd.read_csv(threshold_path)
    else:
        log(f"{run}: sweeping thresholds")
        thdf = sweep_thresholds(vs, evaluator=val_evaluator)
        for col, val in [("run", run), ("group", spec.get("group", "")),
                         ("num_features", len(features)), ("features", ",".join(features))]:
            if col not in thdf.columns:
                thdf.insert(0, col, val)
        thdf.to_csv(threshold_path, index=False)

    selected_path = run_dir / "selected_thresholds.csv"
    if selected_path.exists():
        log(f"RESUME {run}: loading cached selected thresholds")
        seldf = pd.read_csv(selected_path)
    else:
        seldf = select_rows(thdf)
        for col, val in [("run", run), ("group", spec.get("group", "")),
                         ("num_features", len(features)), ("features", ",".join(features))]:
            if col not in seldf.columns:
                seldf.insert(0, col, val)
        seldf.to_csv(selected_path, index=False)

    if len(seldf) == 0:
        raise RuntimeError(f"No selected threshold rows for {run}")

    # LEAK FIX: the mask, stored threshold, and plotted result are now chosen by
    # validation only (best val ESA F0.5, val rate <= 1%). Previously this block
    # sorted validation-selected candidates by test_esa_f05 -- a test-set leak.
    prow = pick_leakfree_row(seldf)
    thr  = float(prow["threshold"])
    mg   = int(prow["merge_gap"])
    md   = int(prow["min_dur"])

    if test_pred_path.exists():
        pred = np.load(test_pred_path)
    else:
        pred = postprocess((ts > thr).astype(np.int8), mg, md)
        np.save(test_pred_path, pred.astype(np.int8))

    if val_pred_path.exists():
        vpred = np.load(val_pred_path)
    else:
        vpred = postprocess((vs > thr).astype(np.int8), mg, md)
        np.save(val_pred_path, vpred.astype(np.int8))

    prow.update({"val_" + k: v for k, v in metrics(yv, vpred, evaluator=val_evaluator).items()})
    prow.update({"test_" + k: v for k, v in metrics(yt, pred, evaluator=test_evaluator).items()})
    pd.DataFrame([prow]).to_csv(selected_path, index=False)

    torch.save({
        "metric_version": METRIC_VERSION,
        "model_state_dict": model.state_dict(),
        "features":    features,
        "scaler_median": scaler.median_,
        "scaler_iqr":    scaler.iqr_,
        "threshold":   thr,
        "merge_gap":   mg,
        "min_dur":     md,
        "selected_row": prow,
        "best_epoch":  best_epoch,
        "best_val":    best_val,
    }, run_dir / "model_checkpoint.pt")

    title = (
        f"{run} | {prow.get('selection_name','')} | "
        f"ESA={float(prow.get('test_esa_f05',0)):.4f} "
        f"event={float(prow.get('test_event_f05',0)):.4f} "
        f"point={float(prow.get('test_point_f05',0)):.4f} "
        f"pred_rate={float(prow.get('test_pred_anomaly_rate',0)):.4f}"
    )

    if not (plot_dir / "overview.jpg").exists():
        log(f"{run}: writing overview plot")
        plot_overview(plot_dir / "overview.jpg", ts, pred, yt, thr, title)

    if not (plot_dir / "per_channel_scores.jpg").exists():
        log(f"{run}: writing per-channel plot")
        plot_channels(plot_dir / "per_channel_scores.jpg", tpc, features, yt, thr, f"Per-channel scores | {run}")

    if not any(plot_dir.glob("event_zoom_*.jpg")):
        log(f"{run}: writing event zoom plots")
        plot_zooms(plot_dir, ts, pred, yt, thr, run)

    summary = {
        "mission": "Mission1", "run": run, "group": spec.get("group", ""),
        "features": features, "num_features": len(features),
        "num_train_rows": int(len(xtr)), "num_val_rows": int(len(yv)),
        "num_test_rows": int(len(yt)),
        "val_true_events": len(val_evaluator.by_id),
        "test_true_events": len(test_evaluator.by_id),
        "evaluation": {"validation": val_evaluator.metadata(), "test": test_evaluator.metadata()},
        "best_epoch": int(best_epoch), "best_val_esa_quick": float(best_val),
        "history": hist, "plot_selection": prow,
        "outputs": {
            "selected_thresholds": str(selected_path.resolve()),
            "threshold_sweep":     str(threshold_path.resolve()),
            "overview_plot":       str((plot_dir / "overview.jpg").resolve()),
            "per_channel_plot":    str((plot_dir / "per_channel_scores.jpg").resolve()),
            "checkpoint_last":     str(checkpoint_last.resolve()),
            "checkpoint_best":     str(checkpoint_best.resolve()),
        },
    }

    save_json(run_dir / "summary.json", summary)
    save_json(state_path, {"run": run, "stage": "completed",
                           "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    # Summary reflects the LEAK-FREE selection. test_esa_f05 is recorded as a
    # measured diagnostic ('_diag'), never as the quantity selection optimised.
    wb.log_summary({
        "selection_name":   prow.get("selection_name"),
        "n_channels":       len(features),
        "threshold":        thr, "merge_gap": mg, "min_dur": md,
        "val_esa_f05":      prow.get("val_esa_f05"),
        "val_pred_rate":    prow.get("val_pred_anomaly_rate"),
        "val_event_f05":    prow.get("val_event_f05"),
        "test_esa_f05_diag":   prow.get("test_esa_f05"),
        "test_event_f05_diag": prow.get("test_event_f05"),
        "best_epoch":       best_epoch,
        "mlp_best_val":     (mlp_best_val if TRAIN_MLP else None),
    })
    wb.finish()

    free_memory(model, opt, sched, xtr, xv, ytr, yv, yt)
    return summary


def combine(out_base: Path):
    selected = [pd.read_csv(p) for p in sorted(out_base.glob("*/selected_thresholds.csv"))]
    if selected:
        frame = pd.concat(selected, ignore_index=True).sort_values("val_esa_f05", ascending=False)
        frame.to_csv(out_base / "selected_thresholds_all_runs.csv", index=False)
        frame.to_csv(out_base / "excel_summary_rows.csv", index=False)
    sweeps = [pd.read_csv(p) for p in sorted(out_base.glob("*/threshold_sweep_val_to_test.csv"))]
    if sweeps:
        pd.concat(sweeps, ignore_index=True).to_csv(out_base / "threshold_sweep_all_runs.csv", index=False)


def write_error(out_base: Path, run: str, e: Exception):
    p = out_base / "errors.csv"; exists = p.exists()
    with open(p, "a", encoding="utf-8") as f:
        if not exists:
            f.write("run,error_type,error,created_at\n")
        msg = str(e).replace("\n"," ").replace("\r"," ").replace(","," ;")
        f.write(f"{run},{type(e).__name__},{msg},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def main():
    set_seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    latest_path = OUT_ROOT / "latest_run.txt"
    out_base    = None

    if latest_path.exists():
        try:
            candidate = Path(latest_path.read_text(encoding="utf-8").strip())
            if candidate.exists():
                out_base = candidate
        except Exception:
            out_base = None

    if out_base is None:
        out_base = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
        out_base.mkdir(parents=True, exist_ok=True)
    else:
        out_base.mkdir(parents=True, exist_ok=True)

    # Fail before updating sweep summaries if any existing run is incompatible.
    for spec in RUN_SPECS:
        existing = out_base / spec["run"]
        if existing.exists():
            guard_run(existing, manifest(spec))

    latest_path.write_text(str(out_base.resolve()) + "\n", encoding="utf-8")
    (out_base / "plots").mkdir(exist_ok=True)

    available = read_columns(TRAIN_FILE)
    missing = {f for spec in RUN_SPECS for f in spec["features"]} - set(available)
    if missing:
        raise ValueError(f"Configured input channels missing: {sorted(missing)}")
    planned   = [{**s, "features": [f for f in s["features"] if f in available]}
                 for s in RUN_SPECS]
    summary   = {
        "run_id": out_base.name, "out_dir": str(out_base.resolve()),
        "mission": "Mission1", "method": "reconstruction_autoencoder_normal_windows",
        "train_file": str(TRAIN_FILE), "test_file": str(TEST_FILE),
        "runs_planned": planned,
        "config": {
            "seq_len": SEQ_LEN, "patch_size": PATCH_SIZE, "stride": STRIDE,
            "score_stride": SCORE_STRIDE, "d_model": D_MODEL, "n_heads": N_HEADS,
            "n_layers": N_LAYERS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "merge_gaps": MERGE_GAPS, "min_durs": MIN_DURS,
        },
        "runs": [], "errors": [],
    }

    log("="*100)
    log("Mission1 reconstruction AE + MLP Forecaster sweep  [OPTIMIZED v2]")
    log(f"device={DEVICE}")
    if torch.cuda.is_available():
        log(f"gpu={torch.cuda.get_device_name(0)}")
    log(f"output={out_base}")
    log(f"planned runs={len(planned)}")
    log(f"MLP forecaster: {'ENABLED' if TRAIN_MLP else 'DISABLED'}  "
        f"(epochs={MLP_EPOCHS}  hidden={MLP_HIDDEN}  layers={MLP_LAYERS})")
    log("resume mode: uses latest_run.txt and cached checkpoints/scores when present")

    for i, s in enumerate(planned, 1):
        if not s["features"]:
            continue
        try:
            log(f"Starting {i}/{len(planned)} {s['run']}")
            res = train_one(s, out_base, available)
            summary["runs"].append(res)
        except torch.cuda.OutOfMemoryError as e:
            log(f"CUDA OOM {s['run']}: {e}")
            write_error(out_base, s["run"], e)
            summary["errors"].append({"run": s["run"], "error_type": "CUDA_OutOfMemoryError", "error": str(e)})
            free_memory(); continue
        except Exception as e:
            log(f"ERROR {s['run']}: {type(e).__name__}: {e}")
            write_error(out_base, s["run"], e)
            summary["errors"].append({"run": s["run"], "error_type": type(e).__name__, "error": str(e)})
            free_memory(); continue
        combine(out_base)
        (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    combine(out_base)
    summary["num_successful_runs"] = len(summary["runs"])
    summary["num_failed_runs"]     = len(summary["errors"])
    summary["outputs"] = {
        "selected_thresholds_all_runs": str((out_base / "selected_thresholds_all_runs.csv").resolve()),
        "threshold_sweep_all_runs":     str((out_base / "threshold_sweep_all_runs.csv").resolve()),
        "excel_summary_rows":           str((out_base / "excel_summary_rows.csv").resolve()),
        "plots_dir":                    str((out_base / "plots").resolve()),
    }
    (out_base / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("="*100)
    log(f"Done  successful={summary['num_successful_runs']}  failed={summary['num_failed_runs']}")
    log(f"saved to {out_base}")

    if (out_base / "excel_summary_rows.csv").exists():
        df   = pd.read_csv(out_base / "excel_summary_rows.csv")
        cols = ["selection_type","selection_name","run","test_esa_f05","test_event_f05",
                "test_point_f05","test_event_precision","test_event_recall","test_pred_anomaly_rate"]
        print(df[[c for c in cols if c in df.columns]].head(30).to_string(index=False), flush=True)

    if summary["num_failed_runs"]:
        raise RuntimeError(f"{summary['num_failed_runs']} training runs failed; inspect errors.csv")
