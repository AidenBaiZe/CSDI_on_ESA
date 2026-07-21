from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset_esa import (  # noqa: E402
    CyclicPermutationSampler,
    build_window_starts,
    deterministic_condition_mask,
    evenly_spaced_indices,
)
from experiment import (  # noqa: E402
    _bucket_masks,
    baseline_predictions,
    sanitize_nonfinite_gradients,
    stable_clip_grad_norm_,
    sorted_linear_quantiles,
)
from preprocess import (  # noqa: E402
    apply_derivative,
    category_code,
    classify_channel,
    compute_normalization,
    read_channel,
    resample_official,
    restoration_points,
)


def test_official_boundary_and_windows_do_not_cross() -> None:
    train_start = pd.Timestamp("2000-01-01T00:00:00")
    split_at = pd.Timestamp("2001-10-01T00:00:00")
    test_end = pd.Timestamp("2003-07-01T00:00:00")
    assert len(pd.date_range(train_start, split_at, freq="18s")) == 3_067_201
    assert len(pd.date_range(split_at, test_end, freq="18s")) == 3_062_401
    counts = np.ones(1000, dtype=np.uint8)
    starts = build_window_starts(counts, 96, 48)
    assert np.all(starts >= 0)
    assert np.all(starts + 96 <= len(counts))
    counts[:] = 0
    assert build_window_starts(counts, 96, 48).size == 0


def test_derivative_applies_to_official_mission2_range_only() -> None:
    index = pd.to_datetime(["2000-01-01 00:00:01", "2000-01-01 00:00:19", "2000-01-01 00:01:10"])
    frame = pd.DataFrame({"value": [2.0, 7.0, 4.0]}, index=index)
    differenced = apply_derivative(frame, 29, 29, 46)
    np.testing.assert_array_equal(differenced["value"], [5.0, -3.0, 0.0])
    boundary = apply_derivative(frame, 46, 29, 46)
    np.testing.assert_array_equal(boundary["value"], [5.0, -3.0, 0.0])
    for outside in (4, 11, 28, 47):
        unchanged = apply_derivative(frame, outside, 29, 46)
        np.testing.assert_array_equal(unchanged["value"], frame["value"])


def test_string_channels_are_factorized_like_the_official_script(tmp_path: Path) -> None:
    index = pd.to_datetime(
        ["2000-01-01 00:00:01", "2000-01-01 00:00:19", "2000-01-01 00:01:10", "2000-01-01 00:02:00"]
    )
    series = pd.Series(["OFF", "ON", "OFF", "STANDBY"], index=index, name="channel_7")
    path = tmp_path / "channel_7.zip"
    series.to_frame().to_pickle(path)
    frame, factorization = read_channel(path, "channel_7")
    expected_codes, _ = pd.factorize(series.to_numpy())
    np.testing.assert_array_equal(frame["value"].to_numpy(), expected_codes.astype(np.float64))
    assert factorization is not None
    assert factorization["unique_count"] == 3
    assert factorization["categories"] == ["OFF", "ON", "STANDBY"]

    numeric = pd.Series([1.5, 2.5], index=index[:2], name="channel_8")
    numeric_path = tmp_path / "channel_8.zip"
    numeric.to_frame().to_pickle(numeric_path)
    numeric_frame, numeric_factorization = read_channel(numeric_path, "channel_8")
    assert numeric_factorization is None
    np.testing.assert_array_equal(numeric_frame["value"].to_numpy(), [1.5, 2.5])


def test_mission2_categories_are_anomaly_and_rare_event_only() -> None:
    assert category_code("Anomaly") == 1
    assert category_code("Rare Event") == 2
    try:
        category_code("Communication Gap")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown Mission 2 categories must fail loudly")


def test_zoh_restores_anomaly_on_the_18_second_grid() -> None:
    frequency = pd.Timedelta(seconds=18)
    index = pd.to_datetime(
        ["2000-01-01 00:00:01", "2000-01-01 00:00:05", "2000-01-01 00:00:12"]
    )
    values = np.asarray([1.0, 9.0, 2.0])
    labels = np.asarray([0, 1, 0], dtype=np.uint8)
    restore_ns, restore_values, restore_labels = restoration_points(index, values, labels, frequency)
    assert pd.Timestamp(restore_ns[0]) == pd.Timestamp("2000-01-01 00:00:18")
    assert restore_values.tolist() == [9.0]
    assert restore_labels.tolist() == [1]

    raw = pd.DataFrame({"value": values, "label": labels}, index=index)
    grid = pd.date_range("2000-01-01 00:00:00", "2000-01-01 00:00:54", freq="18s")
    result_values, result_labels, update, metadata = resample_official(raw, grid, frequency)
    assert result_values[1] == 9.0
    assert result_labels[1] == 1
    assert update[1] == 1
    assert metadata["restored_annotation_points"] == 1


def test_global_bfill_is_observed_held_and_update_semantics() -> None:
    frequency = pd.Timedelta(seconds=18)
    index = pd.to_datetime(["2000-01-01 00:00:36", "2000-01-01 00:00:48"])
    raw = pd.DataFrame(
        {"value": [4.0, 5.0], "label": np.asarray([0, 0], dtype=np.uint8)}, index=index
    )
    grid = pd.date_range("2000-01-01 00:00:00", "2000-01-01 00:01:12", freq="18s")
    values, labels, update, metadata = resample_official(raw, grid, frequency)
    assert values[:2].tolist() == [4.0, 4.0]
    assert labels[:2].tolist() == [0, 0]
    assert update[:2].tolist() == [0, 0]
    assert metadata["leading_bfill_points"] == 2
    assert update[2] == 1


def test_clean_only_normalization_near_constant_still_subtracts_mean() -> None:
    clean = np.asarray([7.0, 7.0, 7.0])
    mean, raw_std, scale, near_constant = compute_normalization(clean, 1e-8)
    assert mean == 7.0
    assert raw_std == 0.0
    assert scale == 1.0
    assert near_constant
    normalized = (np.asarray([7.0, 8.0]) - mean) / scale
    np.testing.assert_array_equal(normalized, [0.0, 1.0])


def test_channel_classification_and_differenced_are_orthogonal() -> None:
    unique_count, discrete, differenced = classify_channel(np.asarray([0, 1, 1]), 29, 64, 29, 46)
    assert unique_count == 2 and discrete and differenced
    _, continuous, not_differenced = classify_channel(np.arange(65), 4, 64, 29, 46)
    assert not continuous and not not_differenced


def test_fixed_eval_masks_are_reproducible_and_independent() -> None:
    observed = np.ones((96, 100), dtype=np.uint8)
    ten_a = deterministic_condition_mask(observed, 0.1, 2101, 480)
    ten_b = deterministic_condition_mask(observed, 0.1, 2101, 480)
    fifty = deterministic_condition_mask(observed, 0.5, 2501, 480)
    np.testing.assert_array_equal(ten_a, ten_b)
    assert int((observed - ten_a).sum()) == round(observed.size * 0.1)
    assert int((observed - fifty).sum()) == round(observed.size * 0.5)
    target_ten = (observed - ten_a).astype(bool)
    target_fifty = (observed - fifty).astype(bool)
    assert not np.all(~target_ten | target_fifty)


def test_training_sampler_is_no_replacement_with_deterministic_reshuffle() -> None:
    sampled = list(CyclicPermutationSampler(7, seed=1, total_samples=16))
    assert len(set(sampled[:7])) == 7
    assert len(set(sampled[7:14])) == 7
    assert sampled == list(CyclicPermutationSampler(7, seed=1, total_samples=16))
    resumed = list(CyclicPermutationSampler(7, seed=1, total_samples=16, start_offset=9))
    assert resumed == sampled[9:]


def test_evenly_spaced_window_selection() -> None:
    selected = evenly_spaced_indices(1000, 256)
    assert selected.size == 256
    assert selected[0] == 0 and selected[-1] == 999
    assert np.unique(selected).size == selected.size


def test_baselines_never_read_masked_values() -> None:
    values = np.asarray([[[1.0, 100.0, 3.0, 200.0, 5.0]]], dtype=np.float32)
    condition = np.asarray([[[1, 0, 1, 0, 1]]], dtype=np.uint8)
    forward_a, linear_a = baseline_predictions(values, condition)
    altered = values.copy()
    altered[0, 0, [1, 3]] = [-9999.0, 9999.0]
    forward_b, linear_b = baseline_predictions(altered, condition)
    np.testing.assert_array_equal(forward_a, forward_b)
    np.testing.assert_array_equal(linear_a, linear_b)
    assert forward_a[0, 0, 1] == 1.0
    assert linear_a[0, 0, 1] == 2.0


def test_metric_bucket_counts_reconcile() -> None:
    target = torch.ones((1, 2, 4))
    labels = torch.tensor([[[0, 1, 2, 0], [0, 0, 1, 2]]])
    update = torch.tensor([[[1, 0, 1, 0], [0, 1, 0, 1]]])
    all_mask, nominal, abnormal, updates, held = _bucket_masks(target, labels, update)
    assert all_mask.sum() == nominal.sum() + abnormal.sum()
    assert all_mask.sum() == updates.sum() + held.sum()


def test_sorted_quantiles_equal_torch_quantile_linear() -> None:
    generator = torch.Generator().manual_seed(7)
    samples = torch.randn((3, 50, 4, 5), generator=generator)
    quantiles = [0.05, 0.5, 0.95]
    actual = sorted_linear_quantiles(samples, quantiles)
    expected = torch.quantile(samples, torch.tensor(quantiles), dim=1, interpolation="linear")
    torch.testing.assert_close(actual, expected)


def test_nonfinite_gradient_sanitizer_changes_only_bad_elements() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[float("nan"), 3.0]])
    assert sanitize_nonfinite_gradients(model) == 1
    torch.testing.assert_close(model.weight.grad, torch.tensor([[0.0, 3.0]]))


def test_stable_gradient_norm_does_not_overflow_on_large_finite_values() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[1.0e30, -1.0e30]])
    norm, fallback = stable_clip_grad_norm_(model, 1.0)
    assert norm is None
    assert fallback
    torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight.grad))
