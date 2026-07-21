import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GPVAE_ROOT = REPO_ROOT / "external" / "GP-VAE"
if str(GPVAE_ROOT) not in sys.path:
    sys.path.insert(0, str(GPVAE_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GP-VAE checkpoint on a CSDI PhysioNet split.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--input-mode",
        choices=["csdi", "zero"],
        default="csdi",
        help="csdi uses x_test_miss from the exported split; zero removes all conditioning input.",
    )
    parser.add_argument("--latent-dim", type=int, default=35)
    parser.add_argument("--encoder-sizes", default="128,128")
    parser.add_argument("--decoder-sizes", default="256,256")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=1.005)
    parser.add_argument("--length-scale", type=float, default=7.0)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--kernel", default="cauchy")
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--inter-op-threads", type=int, default=2)
    parser.add_argument("--omp-num-threads", type=int, default=4)
    return parser.parse_args()


def calc_quantile_crps_points(target, forecast):
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    if denom == 0:
        raise ValueError("CRPS denominator is zero")

    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(forecast, q, axis=1)
        q_loss = 2.0 * np.sum(
            np.abs((q_pred - target) * ((target <= q_pred).astype(np.float32) - q))
        )
        crps += q_loss / denom
    return float(crps / len(quantiles))


def main():
    args = parse_args()
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(args.intra_op_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(args.inter_op_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    import tensorflow as tf

    tf.compat.v1.enable_eager_execution()
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    from lib.models import BandedJointEncoder, GP_VAE, GaussianDecoder

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "imputed_samples_no_gt.npy"

    data_file = Path(args.data_file).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    data = np.load(data_file)
    x_test_miss = data["x_test_miss"].astype(np.float32)
    if args.input_mode == "zero":
        x_test_miss = np.zeros_like(x_test_miss, dtype=np.float32)
    target = data["x_test_full"].astype(np.float32)
    eval_bool = data["m_test_artificial"].astype(bool)

    encoder_sizes = [int(x) for x in args.encoder_sizes.split(",") if x]
    decoder_sizes = [int(x) for x in args.decoder_sizes.split(",") if x]
    model = GP_VAE(
        latent_dim=args.latent_dim,
        data_dim=35,
        time_length=48,
        encoder_sizes=encoder_sizes,
        encoder=BandedJointEncoder,
        decoder_sizes=decoder_sizes,
        decoder=GaussianDecoder,
        kernel=args.kernel,
        sigma=args.sigma,
        length_scale=args.length_scale,
        kernel_scales=1,
        image_preprocessor=None,
        window_size=args.window_size,
        beta=args.beta,
        M=1,
        K=1,
        data_type="physionet",
    )
    _ = tf.compat.v1.train.get_or_create_global_step()
    _ = model.get_trainable_vars()
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    saver = tf.compat.v1.train.Checkpoint(
        optimizer=optimizer,
        encoder=model.encoder.net,
        decoder=model.decoder.net,
        optimizer_step=tf.compat.v1.train.get_or_create_global_step(),
    )
    checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    saver.restore(checkpoint).expect_partial()

    samples = []
    for i in range(args.num_samples):
        batch_samples = []
        for batch in np.array_split(x_test_miss, args.batch_size, axis=0):
            z = model.encode(batch).sample()
            batch_samples.append(model.decode(z).sample().numpy())
        samples.append(np.vstack(batch_samples).astype(np.float32))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"sample {i + 1}/{args.num_samples}", flush=True)
    samples = np.stack(samples, axis=1)
    np.save(samples_path, samples)

    sample_points = np.transpose(samples, (0, 2, 3, 1))[eval_bool]
    target_points = target[eval_bool]
    median = np.median(sample_points, axis=1)
    errors = np.abs(median - target_points)
    sq_errors = (median - target_points) ** 2
    metrics = {
        "status": "completed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint": checkpoint,
        "data_file": str(data_file),
        "samples_path": str(samples_path),
        "test": {
            "mae": float(errors.mean()),
            "mse": float(sq_errors.mean()),
            "crps": calc_quantile_crps_points(target_points, sample_points),
            "eval_points": int(eval_bool.sum()),
            "num_samples": int(samples.shape[1]),
        },
        "args": vars(args),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "gpu_available": bool(tf.test.is_gpu_available()),
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
