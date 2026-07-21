from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


QUANTILES = tuple(float(value) for value in np.arange(0.05, 1.0, 0.05))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def train_model(
    model: torch.nn.Module,
    train_loader: Iterable,
    valid_loader: Iterable,
    train_config: dict[str, Any],
    output_dir: Path,
    epochs: int,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = Adam(model.parameters(), lr=float(train_config["lr"]), weight_decay=1e-6)
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    milestones = sorted({value for value in (p1, p2) if value > 0})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )
    valid_interval = int(train_config.get("valid_epoch_interval", 20))
    history: list[dict[str, Any]] = []
    best_valid = math.inf
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_total = 0.0
        train_batches = 0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{epochs}", mininterval=2.0)
        for batch_index, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_batches += 1
            progress.set_postfix(loss=train_total / train_batches)
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            if batch_index >= float(train_config.get("itr_per_epoch", 1.0e8)):
                break
        scheduler.step()

        valid_loss = None
        should_validate = (epoch + 1) % valid_interval == 0 or epoch + 1 == epochs
        if should_validate:
            model.eval()
            valid_total = 0.0
            valid_count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(valid_loader, start=1):
                    loss = model(batch, is_train=0)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite validation loss at epoch {epoch + 1}"
                        )
                    valid_total += float(loss.item())
                    valid_count += 1
                    if max_valid_batches is not None and batch_index >= max_valid_batches:
                        break
            valid_loss = valid_total / max(valid_count, 1)
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save(model.state_dict(), output_dir / "model_best.pth")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_total / max(train_batches, 1),
            "valid_loss": valid_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.perf_counter() - epoch_start,
            "train_batches": train_batches,
        }
        history.append(epoch_record)
        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record) + "\n")

    torch.save(model.state_dict(), output_dir / "model_final.pth")
    duration = time.perf_counter() - start_time
    summary = {
        "epochs": epochs,
        "duration_seconds": duration,
        "seconds_per_epoch": duration / epochs,
        "best_validation_loss": None if math.isinf(best_valid) else best_valid,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "history": history,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def _quantile_loss_sum(
    target: torch.Tensor,
    forecast: torch.Tensor,
    quantile: float,
    eval_points: torch.Tensor,
) -> torch.Tensor:
    return 2.0 * torch.sum(
        torch.abs((forecast - target) * eval_points)
        * ((target <= forecast).float() - quantile).abs()
    )


def impute_chunked(
    model: torch.nn.Module,
    observed_data: torch.Tensor,
    cond_mask: torch.Tensor,
    side_info: torch.Tensor,
    n_samples: int,
    sample_chunk_size: int,
) -> torch.Tensor:
    if model.is_unconditional:
        raise ValueError("Chunked evaluator currently supports conditional CSDI only")
    if n_samples <= 0 or sample_chunk_size <= 0:
        raise ValueError("n_samples and sample_chunk_size must be positive")
    batch, channels, length = observed_data.shape
    chunks = []
    generated = 0
    while generated < n_samples:
        chunk = min(sample_chunk_size, n_samples - generated)
        repeated_observed = observed_data.repeat_interleave(chunk, dim=0)
        repeated_mask = cond_mask.repeat_interleave(chunk, dim=0)
        repeated_side = side_info.repeat_interleave(chunk, dim=0)
        current = torch.randn(
            batch * chunk,
            channels,
            length,
            device=observed_data.device,
            dtype=observed_data.dtype,
        )
        for step in range(model.num_steps - 1, -1, -1):
            conditioned = (repeated_mask * repeated_observed).unsqueeze(1)
            noisy_target = ((1.0 - repeated_mask) * current).unsqueeze(1)
            diffusion_input = torch.cat([conditioned, noisy_target], dim=1)
            step_tensor = torch.tensor([step], device=observed_data.device)
            predicted = model.diffmodel(diffusion_input, repeated_side, step_tensor)
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


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    nsample: int,
    means: np.ndarray,
    stds: np.ndarray,
    sample_chunk_size: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    device = model.device
    mean_tensor = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, -1, 1)
    std_tensor = torch.as_tensor(stds, dtype=torch.float32, device=device).view(1, -1, 1)
    mse_norm = mae_norm = mse_original = mae_original = 0.0
    eval_count = 0.0
    per_channel_sq = np.zeros(len(means), dtype=np.float64)
    per_channel_abs = np.zeros(len(means), dtype=np.float64)
    per_channel_count = np.zeros(len(means), dtype=np.float64)
    crps_norm_losses = np.zeros(len(QUANTILES), dtype=np.float64)
    crps_original_losses = np.zeros(len(QUANTILES), dtype=np.float64)
    crps_norm_denom = 0.0
    crps_original_denom = 0.0
    start_time = time.perf_counter()
    processed_batches = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        progress = tqdm(loader, desc="evaluate", mininterval=2.0)
        for batch_index, batch in enumerate(progress, start=1):
            (
                target,
                observed_mask,
                observed_time,
                gt_mask,
                _,
                cut_length,
            ) = model.process_data(batch)
            cond_mask = gt_mask
            eval_points = observed_mask - cond_mask
            side_info = model.get_side_info(observed_time, cond_mask)
            samples = impute_chunked(
                model,
                target,
                cond_mask,
                side_info,
                nsample,
                sample_chunk_size,
            )
            for row in range(len(cut_length)):
                eval_points[row, ..., : cut_length[row].item()] = 0
            median = samples.median(dim=1).values
            error = (median - target) * eval_points
            mse_norm += float((error**2).sum().item())
            mae_norm += float(error.abs().sum().item())
            count = float(eval_points.sum().item())
            eval_count += count

            original_target = target * std_tensor + mean_tensor
            original_samples = samples * std_tensor.unsqueeze(1) + mean_tensor.unsqueeze(1)
            original_median = median * std_tensor + mean_tensor
            original_error = (original_median - original_target) * eval_points
            mse_original += float((original_error**2).sum().item())
            mae_original += float(original_error.abs().sum().item())

            channel_dims = (0, 2)
            per_channel_sq += (original_error**2).sum(dim=channel_dims).cpu().numpy()
            per_channel_abs += original_error.abs().sum(dim=channel_dims).cpu().numpy()
            per_channel_count += eval_points.sum(dim=channel_dims).cpu().numpy()

            crps_norm_denom += float(torch.sum(torch.abs(target * eval_points)).item())
            crps_original_denom += float(
                torch.sum(torch.abs(original_target * eval_points)).item()
            )
            for index, quantile in enumerate(QUANTILES):
                forecast = torch.quantile(samples, quantile, dim=1)
                original_forecast = torch.quantile(original_samples, quantile, dim=1)
                crps_norm_losses[index] += float(
                    _quantile_loss_sum(target, forecast, quantile, eval_points).item()
                )
                crps_original_losses[index] += float(
                    _quantile_loss_sum(
                        original_target, original_forecast, quantile, eval_points
                    ).item()
                )

            processed_batches += 1
            progress.set_postfix(
                rmse=math.sqrt(mse_norm / max(eval_count, 1.0)),
                mae=mae_norm / max(eval_count, 1.0),
            )
            if max_batches is not None and batch_index >= max_batches:
                break

    if eval_count == 0:
        raise ValueError("Evaluation mask contains no target points")
    duration = time.perf_counter() - start_time
    channel_metrics = []
    for index in range(len(means)):
        count = max(per_channel_count[index], 1.0)
        channel_metrics.append(
            {
                "channel_index": index,
                "rmse_original": math.sqrt(per_channel_sq[index] / count),
                "mae_original": per_channel_abs[index] / count,
                "eval_points": int(per_channel_count[index]),
            }
        )
    return {
        "nsample": int(nsample),
        "sample_chunk_size": int(sample_chunk_size),
        "evaluated_batches": processed_batches,
        "eval_points": int(eval_count),
        "rmse_normalized": math.sqrt(mse_norm / eval_count),
        "mae_normalized": mae_norm / eval_count,
        "crps_normalized": float(
            crps_norm_losses.sum() / max(crps_norm_denom, 1e-12) / len(QUANTILES)
        ),
        "rmse_original": math.sqrt(mse_original / eval_count),
        "mae_original": mae_original / eval_count,
        "crps_original": float(
            crps_original_losses.sum()
            / max(crps_original_denom, 1e-12)
            / len(QUANTILES)
        ),
        "duration_seconds": duration,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "per_channel": channel_metrics,
    }
