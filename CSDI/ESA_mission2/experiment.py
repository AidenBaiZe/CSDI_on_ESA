from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


BUCKET_NAMES = ("all", "nominal", "anomaly_rare", "update", "held")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def sanitize_nonfinite_gradients(model: torch.nn.Module) -> int:
    """Replace only NaN/Inf gradient elements; return the exact replacement count."""
    replaced = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            finite = torch.isfinite(parameter.grad)
            count = int((~finite).sum().item())
            if count:
                parameter.grad.masked_fill_(~finite, 0.0)
                replaced += count
    return replaced


def stable_clip_grad_norm_(
    model: torch.nn.Module, max_norm: float
) -> tuple[float | None, bool]:
    """Fast clipping; an overflowed aggregate norm safely produces zero gradients."""
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, error_if_nonfinite=False
    )
    overflowed = not bool(torch.isfinite(norm).item())
    return (None if overflowed else float(norm.item())), overflowed


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    samples_seen: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "step": int(step),
            "samples_seen": int(samples_seen),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_state": _rng_state(),
            "history": history,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=model.device, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        _restore_rng_state(payload["rng_state"])
    return payload


def train_fixed_steps(
    model: torch.nn.Module,
    loader: Iterable,
    config: dict[str, Any],
    output_dir: Path,
    start_step: int = 0,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = int(config["steps"])
    batch_size = int(config["batch_size"])
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        payload = load_checkpoint(resume_checkpoint, model, optimizer)
        start_step = int(payload["step"])
        history = list(payload.get("history", []))
    milestones = {int(value) for value in config["lr_milestones"]}
    gradient_clip_norm = float(config.get("gradient_clip_norm", 0.0))
    sanitize_gradients = bool(config.get("sanitize_nonfinite_gradients", False))
    checkpoint_interval = int(config["checkpoint_interval"])
    log_interval = int(config.get("log_interval", 1))
    if start_step >= total_steps:
        raise ValueError(f"Checkpoint step {start_step} already reaches total {total_steps}")
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    interval_started = started
    interval_loss = 0.0
    interval_count = 0
    progress = tqdm(
        loader,
        total=total_steps - start_step,
        desc=f"train {start_step}->{total_steps}",
        mininterval=2.0,
    )
    completed_step = start_step
    for batch in progress:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, is_train=1)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {completed_step + 1}")
        loss.backward()
        nonfinite_gradient_elements = (
            sanitize_nonfinite_gradients(model) if sanitize_gradients else 0
        )
        gradient_norm = None
        gradient_norm_fallback = False
        if gradient_clip_norm > 0:
            gradient_norm, gradient_norm_fallback = stable_clip_grad_norm_(
                model, gradient_clip_norm
            )
        optimizer.step()
        completed_step += 1
        if completed_step in milestones:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        loss_value = float(loss.item())
        interval_loss += loss_value
        interval_count += 1
        if completed_step % log_interval == 0 or completed_step == total_steps:
            record = {
                "step": completed_step,
                "loss": loss_value,
                "interval_mean_loss": interval_loss / interval_count,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm_before_clipping": gradient_norm,
                "gradient_clip_norm": gradient_clip_norm or None,
                "gradient_norm_stable_fallback": gradient_norm_fallback,
                "nonfinite_gradient_elements_replaced": nonfinite_gradient_elements,
                "step_seconds": time.perf_counter() - step_started,
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_jsonl(output_dir / "train_log.jsonl", record)
            history.append(record)
            interval_loss = 0.0
            interval_count = 0
            interval_started = time.perf_counter()
        progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")
        if completed_step % checkpoint_interval == 0 or completed_step == total_steps:
            save_checkpoint(
                output_dir / f"checkpoint_step_{completed_step:06d}.pth",
                model,
                optimizer,
                completed_step,
                completed_step * batch_size,
                history,
            )
        if completed_step >= total_steps:
            break
    duration = time.perf_counter() - started
    if completed_step != total_steps:
        raise RuntimeError(f"Training stopped at {completed_step}, expected {total_steps}")
    torch.save(
        model.state_dict(), output_dir / f"model_final_step_{completed_step:06d}.pth"
    )
    summary = {
        "start_step": int(start_step),
        "final_step": completed_step,
        "batch_size": batch_size,
        "samples_seen_total": completed_step * batch_size,
        "duration_seconds_this_run": duration,
        "seconds_per_step_this_run": duration / max(completed_step - start_step, 1),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if torch.cuda.is_available()
            else 0
        ),
        "fixed_final_checkpoint": str(
            (output_dir / f"checkpoint_step_{completed_step:06d}.pth").resolve()
        ),
        "validation_used": False,
        "best_checkpoint_selection_used": False,
        "gradient_clip_norm": gradient_clip_norm or None,
        "sanitize_nonfinite_gradients": sanitize_gradients,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def sorted_linear_quantiles(
    samples: torch.Tensor, quantiles: list[float] | tuple[float, ...], presorted: bool = False
) -> torch.Tensor:
    """torch.quantile-compatible linear interpolation over sample dimension 1."""
    if samples.ndim < 2 or samples.shape[1] < 1:
        raise ValueError("samples must have a non-empty sample dimension at dim=1")
    sorted_samples = samples if presorted else torch.sort(samples, dim=1).values
    count = sorted_samples.shape[1]
    forecasts = []
    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantiles must be in [0, 1]")
        position = float(quantile) * (count - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        forecast = sorted_samples[:, lower]
        if upper != lower:
            forecast = forecast + (sorted_samples[:, upper] - forecast) * weight
        forecasts.append(forecast)
    return torch.stack(forecasts, dim=0)


def impute_chunked(
    model: torch.nn.Module,
    observed_data: torch.Tensor,
    cond_mask: torch.Tensor,
    side_info: torch.Tensor,
    n_samples: int,
    sample_chunk_size: int,
) -> torch.Tensor:
    if model.is_unconditional:
        raise ValueError("This experiment requires conditional CSDI")
    if n_samples <= 0 or sample_chunk_size <= 0:
        raise ValueError("n_samples and sample_chunk_size must be positive")
    batch, channels, length = observed_data.shape
    generated = 0
    chunks = []
    while generated < n_samples:
        chunk = min(sample_chunk_size, n_samples - generated)
        repeated_observed = observed_data.repeat_interleave(chunk, dim=0)
        repeated_mask = cond_mask.repeat_interleave(chunk, dim=0)
        repeated_side = side_info.repeat_interleave(chunk, dim=0)
        current = torch.randn(
            batch * chunk,
            channels,
            length,
            dtype=observed_data.dtype,
            device=observed_data.device,
        )
        for step in range(model.num_steps - 1, -1, -1):
            conditioned = (repeated_mask * repeated_observed).unsqueeze(1)
            noisy_target = ((1.0 - repeated_mask) * current).unsqueeze(1)
            diffusion_input = torch.cat([conditioned, noisy_target], dim=1)
            predicted = model.diffmodel(
                diffusion_input,
                repeated_side,
                torch.tensor([step], dtype=torch.long, device=observed_data.device),
            )
            coeff1 = 1.0 / model.alpha_hat[step] ** 0.5
            coeff2 = (1.0 - model.alpha_hat[step]) / (1.0 - model.alpha[step]) ** 0.5
            current = coeff1 * (current - coeff2 * predicted)
            if step > 0:
                sigma = (
                    (1.0 - model.alpha[step - 1])
                    / (1.0 - model.alpha[step])
                    * model.beta[step]
                ) ** 0.5
                current += sigma * torch.randn_like(current)
        chunks.append(current.reshape(batch, chunk, channels, length))
        generated += chunk
    return torch.cat(chunks, dim=1)


def baseline_predictions(
    values: np.ndarray, condition_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Leak-free LOCF and linear interpolation in normalized space (fallback mean=0)."""
    values = np.asarray(values, dtype=np.float32)
    visible = np.asarray(condition_mask, dtype=bool)
    if values.shape != visible.shape or values.ndim != 3:
        raise ValueError("values and condition_mask must have shape [batch, channel, time]")
    batch, channels, length = values.shape
    forward = np.zeros_like(values)
    linear = np.zeros_like(values)
    time_index = np.arange(length)
    for row in range(batch):
        for channel in range(channels):
            known = np.flatnonzero(visible[row, channel])
            if known.size == 0:
                continue
            known_values = values[row, channel, known]
            linear[row, channel] = np.interp(time_index, known, known_values)
            last = 0.0
            known_pointer = 0
            for position in range(length):
                if known_pointer < known.size and known[known_pointer] == position:
                    last = float(known_values[known_pointer])
                    known_pointer += 1
                forward[row, channel, position] = last
    return forward, linear


class MetricAccumulator:
    def __init__(self, channel_names: tuple[str, ...], quantiles: list[float]) -> None:
        self.channel_names = channel_names
        self.quantiles = quantiles
        shape = (len(BUCKET_NAMES), len(channel_names))
        self.count = np.zeros(shape, dtype=np.int64)
        self.sse: dict[tuple[str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(shape, dtype=np.float64)
        )
        self.sae: dict[tuple[str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(shape, dtype=np.float64)
        )
        self.pinball: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(shape, dtype=np.float64)
        )
        self.abs_target: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(shape, dtype=np.float64)
        )

    def add_counts(self, bucket_masks: list[torch.Tensor]) -> None:
        for bucket_index, mask in enumerate(bucket_masks):
            self.count[bucket_index] += mask.sum(dim=(0, 2)).cpu().numpy().astype(np.int64)

    def add_errors(
        self,
        method: str,
        scale: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        bucket_masks: list[torch.Tensor],
    ) -> None:
        error = prediction - target
        for bucket_index, mask in enumerate(bucket_masks):
            self.sse[(method, scale)][bucket_index] += (
                error.square() * mask
            ).sum(dim=(0, 2)).cpu().numpy()
            self.sae[(method, scale)][bucket_index] += (
                error.abs() * mask
            ).sum(dim=(0, 2)).cpu().numpy()

    def add_crps(
        self,
        scale: str,
        quantile_forecasts: torch.Tensor,
        target: torch.Tensor,
        bucket_masks: list[torch.Tensor],
    ) -> None:
        for bucket_index, mask in enumerate(bucket_masks):
            self.abs_target[scale][bucket_index] += (
                target.abs() * mask
            ).sum(dim=(0, 2)).cpu().numpy()
        for quantile_index, quantile in enumerate(self.quantiles):
            forecast = quantile_forecasts[quantile_index]
            residual = target - forecast
            loss = 2.0 * torch.maximum(
                quantile * residual, (quantile - 1.0) * residual
            )
            for bucket_index, mask in enumerate(bucket_masks):
                self.pinball[scale][bucket_index] += (
                    loss * mask
                ).sum(dim=(0, 2)).cpu().numpy()

    def rows(
        self,
        ratio: float,
        channel_metadata: tuple[dict, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        methods = ("csdi", "forward_fill", "linear_interpolation")
        for method in methods:
            for scale in ("normalized", "original"):
                for bucket_index, bucket in enumerate(BUCKET_NAMES):
                    for channel_index, channel_name in enumerate(self.channel_names):
                        count = int(self.count[bucket_index, channel_index])
                        sse = self.sse[(method, scale)][bucket_index, channel_index]
                        sae = self.sae[(method, scale)][bucket_index, channel_index]
                        row = {
                            "missing_ratio": ratio,
                            "method": method,
                            "scale": scale,
                            "bucket": bucket,
                            "channel": channel_name,
                            "eval_points": count,
                            "rmse": math.sqrt(sse / count) if count else None,
                            "mae": sae / count if count else None,
                            "crps": None,
                            "crps_scaled_by_abs_target": None,
                            "value_type": channel_metadata[channel_index]["value_type"],
                            "near_constant": channel_metadata[channel_index]["near_constant"],
                            "differenced": channel_metadata[channel_index]["differenced"],
                        }
                        if method == "csdi" and count:
                            pinball = self.pinball[scale][bucket_index, channel_index]
                            row["crps"] = pinball / (len(self.quantiles) * count)
                            absolute = self.abs_target[scale][bucket_index, channel_index]
                            if absolute > 0:
                                row["crps_scaled_by_abs_target"] = (
                                    pinball / (len(self.quantiles) * absolute)
                                )
                        rows.append(row)
        return rows


def _bucket_masks(
    target_mask: torch.Tensor, label_code: torch.Tensor, update_mask: torch.Tensor
) -> list[torch.Tensor]:
    target = target_mask.float()
    return [
        target,
        target * (label_code == 0),
        target * ((label_code == 1) | (label_code == 2)),
        target * (update_mask == 1),
        target * (update_mask == 0),
    ]


def aggregate_metric_rows(
    channel_rows: list[dict[str, Any]], channel_metadata: tuple[dict, ...]
) -> list[dict[str, Any]]:
    groups = {
        "all_channels": np.ones(len(channel_metadata), dtype=bool),
        "continuous": np.asarray([m["value_type"] == "continuous" for m in channel_metadata]),
        "discrete_like": np.asarray([m["value_type"] == "discrete_like" for m in channel_metadata]),
        "near_constant": np.asarray([bool(m["near_constant"]) for m in channel_metadata]),
        "differenced": np.asarray([bool(m["differenced"]) for m in channel_metadata]),
        "non_differenced": np.asarray([not bool(m["differenced"]) for m in channel_metadata]),
    }
    channel_index = {f"channel_{m['channel_id']}": index for index, m in enumerate(channel_metadata)}
    result = []
    keys = sorted(
        {(r["missing_ratio"], r["method"], r["scale"], r["bucket"]) for r in channel_rows}
    )
    lookup = {
        (r["missing_ratio"], r["method"], r["scale"], r["bucket"], r["channel"]): r
        for r in channel_rows
    }
    for ratio, method, scale, bucket in keys:
        for group_name, mask in groups.items():
            selected = [index for index, enabled in enumerate(mask) if enabled]
            rows = [
                lookup[(ratio, method, scale, bucket, f"channel_{channel_metadata[i]['channel_id']}")]
                for i in selected
            ]
            count = sum(r["eval_points"] for r in rows)
            if count:
                sse = sum((r["rmse"] ** 2) * r["eval_points"] for r in rows if r["rmse"] is not None)
                sae = sum(r["mae"] * r["eval_points"] for r in rows if r["mae"] is not None)
                crps_num = sum(
                    r["crps"] * r["eval_points"] for r in rows if r["crps"] is not None
                )
                crps_count = sum(r["eval_points"] for r in rows if r["crps"] is not None)
            else:
                sse = sae = crps_num = crps_count = 0
            result.append(
                {
                    "missing_ratio": ratio,
                    "method": method,
                    "scale": scale,
                    "bucket": bucket,
                    "group": group_name,
                    "channels": len(selected),
                    "eval_points": count,
                    "rmse": math.sqrt(sse / count) if count else None,
                    "mae": sae / count if count else None,
                    "crps": crps_num / crps_count if crps_count else None,
                }
            )
    return result


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    ratio: float,
    nsample: int,
    sample_chunk_size: int,
    means: np.ndarray,
    scales: np.ndarray,
    channel_names: tuple[str, ...],
    channel_metadata: tuple[dict, ...],
    quantiles: list[float],
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    device = model.device
    mean = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, -1, 1)
    scale = torch.as_tensor(scales, dtype=torch.float32, device=device).view(1, -1, 1)
    accumulator = MetricAccumulator(channel_names, quantiles)
    processed_batches = 0
    processed_windows = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        progress = tqdm(loader, desc=f"evaluate missing={ratio:.0%}", mininterval=2.0)
        for batch_index, batch in enumerate(progress, start=1):
            target, observed, timepoints, condition, _, _ = model.process_data(batch)
            target_mask = observed - condition
            if target_mask.sum() <= 0:
                raise ValueError("Evaluation batch has no artificially masked targets")
            label = batch["label_code"].to(device).permute(0, 2, 1)
            update = batch["update_mask"].to(device).permute(0, 2, 1)
            buckets = _bucket_masks(target_mask, label, update)
            accumulator.add_counts(buckets)
            side_info = model.get_side_info(timepoints, condition)
            samples = impute_chunked(
                model, target, condition, side_info, nsample, sample_chunk_size
            )
            sorted_samples = torch.sort(samples, dim=1).values
            quantile_forecasts = sorted_linear_quantiles(
                sorted_samples, quantiles, presorted=True
            )
            median = sorted_linear_quantiles(
                sorted_samples, [0.5], presorted=True
            )[0]
            target_original = target * scale + mean
            median_original = median * scale + mean
            quantiles_original = quantile_forecasts * scale.unsqueeze(0) + mean.unsqueeze(0)

            forward_np, linear_np = baseline_predictions(
                target.detach().cpu().numpy(), condition.detach().cpu().numpy()
            )
            forward = torch.from_numpy(forward_np).to(device)
            linear = torch.from_numpy(linear_np).to(device)
            forward_original = forward * scale + mean
            linear_original = linear * scale + mean

            for method, prediction in (
                ("csdi", median),
                ("forward_fill", forward),
                ("linear_interpolation", linear),
            ):
                accumulator.add_errors(method, "normalized", prediction, target, buckets)
            for method, prediction in (
                ("csdi", median_original),
                ("forward_fill", forward_original),
                ("linear_interpolation", linear_original),
            ):
                accumulator.add_errors(method, "original", prediction, target_original, buckets)
            accumulator.add_crps("normalized", quantile_forecasts, target, buckets)
            accumulator.add_crps("original", quantiles_original, target_original, buckets)
            processed_batches += 1
            processed_windows += int(target.shape[0])
            if max_batches is not None and batch_index >= max_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    rows = accumulator.rows(ratio, channel_metadata)
    groups = aggregate_metric_rows(rows, channel_metadata)
    all_norm = next(
        row
        for row in groups
        if row["method"] == "csdi"
        and row["scale"] == "normalized"
        and row["bucket"] == "all"
        and row["group"] == "all_channels"
    )
    return {
        "missing_ratio": ratio,
        "nsample": nsample,
        "sample_chunk_size": sample_chunk_size,
        "processed_batches": processed_batches,
        "processed_windows": processed_windows,
        "duration_seconds": duration,
        "seconds_per_window": duration / processed_windows,
        "projected_seconds_for_256_windows": duration / processed_windows * 256,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if torch.cuda.is_available()
            else 0
        ),
        "headline_normalized": all_norm,
        "channel_rows": rows,
        "group_rows": groups,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
