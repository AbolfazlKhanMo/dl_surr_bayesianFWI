import os
import numpy as np
from pathlib import Path

from config import BASE_DATA, MONITOR_DATA, TARGET_DATA, sigma


def generate_target_from_baseline_and_monitor(
    base_dir,
    monitor_dir,
    target_dir,
    sigma_percent=0.0,
    seed=None,
    meta_filename="data.noise.npz",
):
    """
    Computes (monitor - baseline) + additive Gaussian noise and saves as data.bin.

    Noise model (standard synthetic FWI practice):
        diff_clean = monitor - baseline
        rms_clean  = RMS(diff_clean)
        sigma_n    = (sigma_percent/100) * rms_clean
        noise ~ N(0, sigma_n^2)
        diff_noisy = diff_clean + noise

    Files written to target_dir:
        - data.bin          : float32 noisy difference vector
        - data.noise.npz    : metadata with sigma_n, rms_clean, sigma_percent, seed, n_samples

    Args:
        base_dir (str|Path): directory containing baseline data.bin
        monitor_dir (str|Path): directory containing monitor data.bin
        target_dir (str|Path): directory to write target data.bin and metadata
        sigma_percent (float): noise level as % of RMS(diff_clean) (e.g., 5.0 for 5%)
        seed (int|None): RNG seed for reproducibility
        meta_filename (str): metadata filename to save in target_dir

    Returns:
        (target_path_str, n_samples_int, sigma_n_float, rms_clean_float, meta_path_str)
    """
    rng = np.random.default_rng(seed)

    base_dir = Path(base_dir)
    monitor_dir = Path(monitor_dir)
    target_dir = Path(target_dir)

    base_path = base_dir / "data.bin"
    monitor_path = monitor_dir / "data.bin"
    target_path = target_dir / "data.bin"
    meta_path = target_dir / meta_filename

    # --- Check existence ---
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {base_path}")
    if not monitor_path.exists():
        raise FileNotFoundError(f"Monitor file not found: {monitor_path}")

    # --- Load data ---
    baseline = np.fromfile(base_path, dtype=np.float32)
    monitor = np.fromfile(monitor_path, dtype=np.float32)

    if baseline.shape != monitor.shape:
        raise ValueError(f"Shape mismatch: baseline {baseline.shape}, monitor {monitor.shape}")

    # --- Compute clean difference ---
    diff_clean = monitor - baseline

    # --- Compute RMS(clean diff) using float64 for numerical stability ---
    rms_clean = float(np.sqrt(np.mean(diff_clean.astype(np.float64) ** 2)))

    # --- Convert percent -> absolute noise std ---
    rel = float(sigma_percent) / 100.0
    sigma_n = float(rel * (rms_clean + 1e-12)) if rel > 0.0 else 1.0  # unit variance for noiseless case

    # --- Add noise (additive Gaussian) ---
    if rel > 0.0:  # check noise percentage, not sigma_n (sigma_n=1.0 is unit variance, not noise)
        noise = rng.normal(0.0, sigma_n, size=diff_clean.shape).astype(np.float32)
        diff_noisy = diff_clean + noise
    else:
        diff_noisy = diff_clean

    # --- Save target data ---
    target_dir.mkdir(parents=True, exist_ok=True)
    diff_noisy.astype(np.float32).tofile(target_path)
    
    # --- Save metadata for consistent misfit ---
    np.savez(
        meta_path,
        sigma_percent=float(sigma_percent),
        rms_clean=float(rms_clean),
        sigma_n=float(sigma_n),
        seed=-1 if seed is None else int(seed),
        n_samples=int(diff_noisy.size),
        dtype="float32",
        note="sigma_n = (sigma_percent/100) * RMS(diff_clean); noise ~ N(0, sigma_n^2)",
    )

    print(f"Saved target data: {target_path} ({diff_noisy.size} samples)")
    print(f"Saved noise meta : {meta_path}")
    print(f"Noise: {sigma_percent:.3f}% of RMS(diff_clean={rms_clean:.6e}) -> sigma_n={sigma_n:.6e}")

    return str(target_path), int(diff_noisy.size), float(sigma_n), float(rms_clean), str(meta_path)


# Example usage
if __name__ == "__main__":
    generate_target_from_baseline_and_monitor(
        base_dir=BASE_DATA,
        monitor_dir=MONITOR_DATA,
        target_dir=TARGET_DATA,
        sigma_percent=sigma,   # e.g. 2.0 means 2% of RMS(diff_clean)
        seed=123,
    )
    os.system(f"python visualize_noise.py")


