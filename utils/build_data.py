from pathlib import Path
import os
import shutil
import numpy as np

def _read_float32_exact(path, n_values):
    """Read exactly n_values float32s; raise if file size doesn't match."""
    expected_bytes = n_values * 4
    size_bytes = os.path.getsize(path)
    if size_bytes != expected_bytes:
        raise ValueError(
            f"{path} has {size_bytes} bytes, expected {expected_bytes} "
            f"(n_values={n_values})."
        )
    arr = np.fromfile(path, dtype=np.float32, count=n_values)
    if arr.size != n_values:
        raise ValueError(f"Read {arr.size} values from {path}, expected {n_values}.")
    return arr

def copy_and_save_data(
    n_shots,
    SPECFEM2D_WORKDIR,
    REF_DATA,
    *,
    n_receivers,      # <— pass this in
    nstep,
    n_sample,
    sigma_percent,
    seed=None,
    flatten_order="C",
):
    """
    Copies Ux/Uz from each run, stacks them by [Ux; Uz] for each shot,
    adds multiplicative Gaussian noise, and saves target_data.bin (float32).
    """
    # Normalize types
    nstep = int(nstep)
    n_sample = int(n_sample)
    n_receivers = int(n_receivers)

    # Derived sizes
    samples_per_trace = nstep // n_sample
    if samples_per_trace * n_sample != nstep:
        raise ValueError(f"nstep ({nstep}) must be divisible by n_sample ({n_sample}).")
    values_per_component = n_receivers * samples_per_trace
    values_per_shot = 2 * values_per_component  # Ux + Uz

    # Prepare dirs
    ref = Path(REF_DATA)
    if ref.exists():
        shutil.rmtree(ref)
    ref.mkdir(parents=True)
    workdir = Path(SPECFEM2D_WORKDIR)

    # Preallocate: rows = Ux then Uz (both flattened receiver-major order), cols = shots
    data_matrix = np.empty((values_per_shot, n_shots), dtype=np.float32)

    rng = np.random.default_rng(seed)

    for i in range(1, n_shots + 1):
        run_id = f"run{str(i).zfill(4)}"
        src_dir = workdir / run_id / "OUTPUT_FILES"
        dst_dir = ref / run_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        ux_src = src_dir / "Ux_file_single_d.bin"
        uz_src = src_dir / "Uz_file_single_d.bin"

        # Copy originals for provenance
        shutil.copy2(ux_src, dst_dir / "Ux_file_single_d.bin")
        shutil.copy2(uz_src, dst_dir / "Uz_file_single_d.bin")

        # Read exactly the expected count
        ux = _read_float32_exact(str(ux_src), values_per_component)
        uz = _read_float32_exact(str(uz_src), values_per_component)

        # Optional: reshape for sanity checks or future processing
        # Each as [n_receivers, samples_per_trace] in receiver-major order
        # ux = ux.reshape(n_receivers, samples_per_trace)
        # uz = uz.reshape(n_receivers, samples_per_trace)

        col = i - 1
        data_matrix[0:values_per_component, col] = ux
        data_matrix[values_per_component:, col] = uz

    # Flatten to 1-D vector (shot-stacked) in requested order
    vector = data_matrix.flatten(order=flatten_order)

    # Apply multiplicative Gaussian noise
    sigma = float(sigma_percent) / 100.0
    noise = rng.normal(0.0, sigma, size=vector.shape).astype(np.float32)
    vector_noisy = vector * (1.0 + noise)
    
    # Save
    target_path = ref / "data.bin"
    vector_noisy.astype(np.float32).tofile(target_path)

    return str(target_path), int(vector_noisy.shape[0])