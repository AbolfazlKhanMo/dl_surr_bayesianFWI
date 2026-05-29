import os
import shutil
import joblib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cnn import CNNRegressor

import logging


def setup_output_dirs():
    base_dir = "OUTPUT_FILES"
    subdirs = ["files", "train", "inference"]

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.makedirs(base_dir)
    for sub in subdirs:
        os.makedirs(os.path.join(base_dir, sub))

    return None


# If adaptive = False:
# train/val/test = 70/20/10
adaptive = True


# ──────────────────────────────────────────────────────────────
#  STRATIFIED SPLITTING USING LABEL QUANTILES
# ──────────────────────────────────────────────────────────────

def stratified_split_by_label(labels, val_frac=0.2, n_bins=10, seed=4):
    n = len(labels)
    rng = np.random.default_rng(seed)

    bin_edges = np.quantile(labels, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(labels, bin_edges[:-1]) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    train_idx, val_idx = [], []

    for b in range(n_bins):
        members = np.where(bin_ids == b)[0]
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * val_frac)))
        val_idx.extend(members[:n_val])
        train_idx.extend(members[n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def stratified_split_3way(labels, val_frac=0.2, test_frac=0.1, n_bins=10, seed=4):
    n = len(labels)
    rng = np.random.default_rng(seed)

    bin_edges = np.quantile(labels, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(labels, bin_edges[:-1]) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    train_idx, val_idx, test_idx = [], [], []

    for b in range(n_bins):
        members = np.where(bin_ids == b)[0]
        rng.shuffle(members)
        n_test = max(1, int(round(len(members) * test_frac)))
        n_val = max(1, int(round(len(members) * val_frac)))
        test_idx.extend(members[:n_test])
        val_idx.extend(members[n_test:n_test + n_val])
        train_idx.extend(members[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────
#  DIAGNOSTICS
# ──────────────────────────────────────────────────────────────

def log_label_diagnostics(y_train, y_val, y_test=None):
    sets = [("train", y_train), ("val", y_val)]
    if y_test is not None:
        sets.append(("test", y_test))

    for name, arr in sets:
        if arr is None:
            continue
        flat = arr.flatten()
        logging.info(
            f"  {name:>6s}: n={len(flat):>5d}  "
            f"min={flat.min():.4f}  q25={np.percentile(flat,25):.4f}  "
            f"median={np.median(flat):.4f}  q75={np.percentile(flat,75):.4f}  "
            f"max={flat.max():.4f}  mean={flat.mean():.4f}  std={flat.std():.4f}"
        )


def plot_split_distributions(y_train, y_val, y_test=None, tag="raw"):
    plt.figure(figsize=(10, 5))
    bins = 50
    plt.hist(y_train.flatten(), bins=bins, alpha=0.5, label=f"Train (n={len(y_train)})", density=True)
    plt.hist(y_val.flatten(), bins=bins, alpha=0.5, label=f"Val (n={len(y_val)})", density=True)
    if y_test is not None:
        plt.hist(y_test.flatten(), bins=bins, alpha=0.5, label=f"Test (n={len(y_test)})", density=True)
    plt.xlabel("Label value")
    plt.ylabel("Density")
    plt.title(f"Label distributions ({tag})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"OUTPUT_FILES/train/label_dist_{tag}.png", dpi=200)
    plt.close()


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    setup_output_dirs()

    # Configure logging AFTER output dirs are created
    logging.basicConfig(
        filename="OUTPUT_FILES/training.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M"
    )

    inputs = np.load("dataset/nn_surr/sobol_input_array.npz")["input_array"][:, 1, :, :]
    labels = np.load("dataset/nn_surr/sobol_label_misfit.npz")["label_misfit"] * -1
    labels = np.abs(labels)

    param_values = np.load("dataset/nn_surr/sobol_input_values.npz")["input_values"]
    logging.info(f"Loaded Sobol parameter values: shape={param_values.shape}")

    inputs = inputs.astype(np.float32)
    labels = labels.astype(np.float32).squeeze()

    n = len(labels)
    logging.info(f"Dataset size: {n}")

    if np.any(labels <= 0):
        raise ValueError(
            f"All labels must be > 0 for log-transform. Found min label = {labels.min()}"
        )

    # ── Dataset diagnostics BEFORE sorting ──
    logging.info("=== RAW LABEL DIAGNOSTICS ===")
    logging.info(
        f"  Labels: min={labels.min():.4f}, max={labels.max():.4f}, "
        f"mean={labels.mean():.4f}, std={labels.std():.4f}, "
        f"median={np.median(labels):.4f}"
    )
    skewness = float(np.mean(((labels - labels.mean()) / labels.std())**3))
    logging.info(f"  Skewness: {skewness:.4f}")
    logging.info(f"  Ratio max/min: {labels.max() / labels.min():.1f}")
    logging.info(f"  Ratio max/median: {labels.max() / np.median(labels):.1f}")

    sort_idx = np.argsort(labels)
    inputs = inputs[sort_idx]
    labels = labels[sort_idx]
    param_values = param_values[sort_idx]

    # ── Adaptive hyperparameters based on dataset size ──
    if n <= 600:
        batch_size = 32
        lr = 5.0e-5
    elif n <= 2000:
        batch_size = 64
        lr = 1.0e-4
    else:
        batch_size = 128
        lr = 2.0e-4

    logging.info(f"Adaptive hyperparams for n={n}: batch_size={batch_size}, lr={lr:.1e}")

    # ──────────────────────────────────────────────────────────
    #  K-FOLD CROSS-VALIDATION
    # ──────────────────────────────────────────────────────────
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_val_losses = []

    logging.info(f"\n=== {n_folds}-FOLD CROSS-VALIDATION ===")

    for fold, (train_idx, val_idx) in enumerate(kf.split(inputs)):
        logging.info(f"\n--- Fold {fold+1}/{n_folds} ---")
        logging.info(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

        x_train_raw = inputs[train_idx]
        y_train_raw = labels[train_idx]
        x_val_raw = inputs[val_idx]
        y_val_raw = labels[val_idx]

        logging.info(f"=== FOLD {fold+1} LABEL DIAGNOSTICS (raw) ===")
        log_label_diagnostics(y_train_raw, y_val_raw)

        # ── Input scaling (fit on this fold's training data) ──
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler_X.fit_transform(
            x_train_raw.reshape(x_train_raw.shape[0], -1)
        ).reshape(x_train_raw.shape)
        x_val_scaled = scaler_X.transform(
            x_val_raw.reshape(x_val_raw.shape[0], -1)
        ).reshape(x_val_raw.shape)

        # ── Label scaling (log + standardise, fit on this fold's training data) ──
        y_train_log = np.log(y_train_raw.reshape(-1, 1).astype(np.float64) + 1e-10)
        y_val_log = np.log(y_val_raw.reshape(-1, 1).astype(np.float64) + 1e-10)

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_log).astype(np.float32)
        y_val_scaled = scaler_y.transform(y_val_log).astype(np.float32)

        X_train = prepare_inputs_for_pytorch(x_train_scaled)
        X_val = prepare_inputs_for_pytorch(x_val_scaled)
        y_train = y_train_scaled.reshape(-1, 1)
        y_val = y_val_scaled.reshape(-1, 1)

        best_val = train_model(
            X_train, y_train, X_val, y_val,
            y_log_mean=scaler_y.mean_[0],
            y_log_std=scaler_y.scale_[0],
            lr=lr,
            batch_size=batch_size,
            fold=fold,
        )
        fold_val_losses.append(best_val)
        logging.info(f"  Fold {fold+1} best val loss: {best_val:.6e}")

    # ── CV summary ──
    fold_val_losses = np.array(fold_val_losses)
    logging.info(f"\n=== K-FOLD CV SUMMARY ===")
    logging.info(f"  Per-fold best val losses: {fold_val_losses}")
    logging.info(f"  Mean: {fold_val_losses.mean():.6e} ± Std: {fold_val_losses.std():.6e}")

    plot_cv_summary(fold_val_losses)

    # ──────────────────────────────────────────────────────────
    #  FINAL TRAINING on full stratified split (for deployment)
    # ──────────────────────────────────────────────────────────
    logging.info("\n=== FINAL TRAINING (full stratified split, for deployment) ===")

    if adaptive:
        train_idx, val_idx = stratified_split_by_label(
            labels, val_frac=0.2, n_bins=10, seed=4
        )
        x_train_raw = inputs[train_idx]
        y_train_raw = labels[train_idx]
        x_val_raw = inputs[val_idx]
        y_val_raw = labels[val_idx]

        logging.info("=== FINAL SPLIT LABEL DIAGNOSTICS (raw) ===")
        log_label_diagnostics(y_train_raw, y_val_raw)
        plot_split_distributions(y_train_raw, y_val_raw, tag="raw")
    else:
        train_idx, val_idx, test_idx = stratified_split_3way(
            labels, val_frac=0.2, test_frac=0.1, n_bins=10, seed=4
        )
        x_train_raw = inputs[train_idx]
        y_train_raw = labels[train_idx]
        x_val_raw = inputs[val_idx]
        y_val_raw = labels[val_idx]
        x_test_raw = inputs[test_idx]
        y_test_raw = labels[test_idx]

        logging.info("=== FINAL SPLIT LABEL DIAGNOSTICS (raw) ===")
        log_label_diagnostics(y_train_raw, y_val_raw, y_test_raw)
        plot_split_distributions(y_train_raw, y_val_raw, y_test_raw, tag="raw")

    # ── Input scaling ──
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler_X.fit_transform(
        x_train_raw.reshape(x_train_raw.shape[0], -1)
    ).reshape(x_train_raw.shape)
    x_val_scaled = scaler_X.transform(
        x_val_raw.reshape(x_val_raw.shape[0], -1)
    ).reshape(x_val_raw.shape)

    if not adaptive:
        x_test_scaled = scaler_X.transform(
            x_test_raw.reshape(x_test_raw.shape[0], -1)
        ).reshape(x_test_raw.shape)

    # ── Label scaling (log + standardise) ──
    y_train_log = np.log(y_train_raw.reshape(-1, 1).astype(np.float64) + 1e-10)
    y_val_log = np.log(y_val_raw.reshape(-1, 1).astype(np.float64) + 1e-10)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_log).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val_log).astype(np.float32)

    if not adaptive:
        y_test_log = np.log(y_test_raw.reshape(-1, 1).astype(np.float64) + 1e-10)
        y_test_scaled = scaler_y.transform(y_test_log).astype(np.float32)

    logging.info("=== FINAL SPLIT LABEL DIAGNOSTICS (scaled) ===")
    if adaptive:
        log_label_diagnostics(y_train_scaled, y_val_scaled)
        plot_split_distributions(y_train_scaled, y_val_scaled, tag="scaled")
    else:
        log_label_diagnostics(y_train_scaled, y_val_scaled, y_test_scaled)
        plot_split_distributions(y_train_scaled, y_val_scaled, y_test_scaled, tag="scaled")

    joblib.dump(scaler_X, "OUTPUT_FILES/files/input_scaler.pkl")
    joblib.dump(scaler_y, "OUTPUT_FILES/files/label_scaler.pkl")
    logging.info("Saved input MinMax scaler and label StandardScaler fitted ONLY on training data.")

    X_train = prepare_inputs_for_pytorch(x_train_scaled)
    X_val = prepare_inputs_for_pytorch(x_val_scaled)

    y_train = y_train_scaled.reshape(-1, 1)
    y_val = y_val_scaled.reshape(-1, 1)

    logging.info(f"Training X shape: {X_train.shape}")
    logging.info(f"Validation X shape: {X_val.shape}")
    logging.info(f"Training y shape: {y_train.shape}")
    logging.info(f"Validation y shape: {y_val.shape}")

    if not adaptive:
        X_test = prepare_inputs_for_pytorch(x_test_scaled)
        y_test = y_test_scaled.reshape(-1, 1)
        logging.info(f"Test X shape: {X_test.shape}")
        logging.info(f"Test y shape: {y_test.shape}")

    np.savez("OUTPUT_FILES/train/train.npz", input=np.array(X_train), label=np.array(y_train))
    np.savez("OUTPUT_FILES/inference/val.npz", input=np.array(X_val), label=np.array(y_val))

    if not adaptive:
        np.savez("OUTPUT_FILES/inference/test.npz", input=np.array(X_test), label=np.array(y_test))

    logging.info("Files saved successfully.")

    train_model(
        X_train, y_train, X_val, y_val,
        y_log_mean=scaler_y.mean_[0],
        y_log_std=scaler_y.scale_[0],
        lr=lr,
        batch_size=batch_size,
        fold=None,
    )

    logging.getLogger().handlers[0].stream.write("\n")
    logging.info("Training completed successfully.")
    return None


def prepare_inputs_for_pytorch(inputs):
    X = inputs

    if X.ndim == 3:
        X = X[:, None, :, :]
    elif X.ndim == 4:
        if X.shape[1] in [1, 3]:
            pass
        elif X.shape[-1] in [1, 3]:
            X = X.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f"Unrecognized input shape {X.shape} for image data.")
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {X.shape}.")

    return X.astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  TRAINING LOOP (shared by CV folds and final training)
# ──────────────────────────────────────────────────────────────

def train_model(x_train, y_train, x_test, y_test,
                y_log_mean, y_log_std,
                lr=5.0e-5,
                batch_size=64,
                num_epochs=350,
                patience=50,
                plot_interval=10,
                alpha=1.0,
                fold=None):

    is_final = fold is None
    tag = "final" if is_final else f"fold{fold+1}"

    logging.getLogger().handlers[0].stream.write("\n\n")
    logging.info(f"-------------> Starting training ({tag}) <-------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_test_t = torch.from_numpy(x_test).float()
    y_test_t = torch.from_numpy(y_test).float()

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNNRegressor().to(device)
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )

    criterion = nn.HuberLoss()

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    if is_final:
        checkpoint_path = "OUTPUT_FILES/files/best_model.pt"
    else:
        checkpoint_path = f"OUTPUT_FILES/files/best_model_{tag}.pt"

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            loss = criterion(preds, yb)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ── Validation with MC Dropout ──
        model.eval()
        model.enable_mc_dropout()

        running_val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)

                loss = criterion(preds, yb)

                running_val_loss += loss.item() * xb.size(0)

        model.disable_mc_dropout()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f"[{tag}] Epoch {epoch+1}/{num_epochs} "
            f"- train_loss: {epoch_train_loss:.6e} "
            f"- val_loss: {epoch_val_loss:.6e} "
            f"- lr: {current_lr:.2e}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(
                f"[{tag}] Early stopping at epoch {epoch+1}, "
                f"best epoch was {best_epoch+1} "
                f"with val_loss = {best_val_loss:.6e}"
            )
            break

        if is_final and (epoch + 1) % plot_interval == 0:
            plot_losses(train_losses, val_losses, epoch + 1)

    np.savez(f"OUTPUT_FILES/train/train_losses_{tag}.npz", train_losses=np.array(train_losses))
    np.savez(f"OUTPUT_FILES/train/val_losses_{tag}.npz", val_losses=np.array(val_losses))

    if is_final:
        plot_losses(train_losses, val_losses, epoch + 1, final=True)
    else:
        plot_losses(train_losses, val_losses, epoch + 1, final=False, tag=tag)

    return best_val_loss


# ──────────────────────────────────────────────────────────────
#  PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_cv_summary(fold_val_losses):
    n_folds = len(fold_val_losses)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_folds + 1), fold_val_losses, color="steelblue", edgecolor="black")
    plt.axhline(fold_val_losses.mean(), color="red", linestyle="--",
                label=f"Mean = {fold_val_losses.mean():.4e}")
    plt.fill_between(
        [0.5, n_folds + 0.5],
        fold_val_losses.mean() - fold_val_losses.std(),
        fold_val_losses.mean() + fold_val_losses.std(),
        color="red", alpha=0.1, label=f"±1 std = {fold_val_losses.std():.4e}"
    )
    plt.xlabel("Fold")
    plt.ylabel("Best Validation Loss")
    plt.title(f"{n_folds}-Fold Cross-Validation Results")
    plt.xticks(range(1, n_folds + 1))
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("OUTPUT_FILES/train/cv_summary.png", dpi=200)
    plt.close()


def plot_losses(train_losses, val_losses, epoch, final=False, tag=None):
    epochs = range(1, len(train_losses) + 1)
    label_tag = f" [{tag}]" if tag else ""

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, "k", label="Training loss", linewidth=2)
    plt.plot(epochs, val_losses, "r--", label="Validation loss")
    title_suffix = " (Final)" if final else f" up to Epoch {epoch}"
    plt.title(f"Training and Validation Loss{title_suffix}{label_tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if tag:
        fname_suffix = f"{tag}_epoch_{epoch}"
    else:
        fname_suffix = "final" if final else f"epoch_{epoch}"
    out_path = f"OUTPUT_FILES/train/train_loss_{fname_suffix}.png"
    plt.savefig(out_path, format="png", dpi=600)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, train_losses, "k", label="Training loss", linewidth=2)
    plt.semilogy(epochs, val_losses, "r--", label="Validation loss")
    title_suffix = " (Final)" if final else f" up to Epoch {epoch}"
    plt.title(f"Semi-Log Training and Validation Loss{title_suffix}{label_tag}")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if tag:
        fname_suffix = f"{tag}_epoch_{epoch}"
    else:
        fname_suffix = "final" if final else f"epoch_{epoch}"
    out_path = f"OUTPUT_FILES/train/semi_log_train_loss_{fname_suffix}.png"
    plt.savefig(out_path, format="png", dpi=600)
    plt.close()

    return None


if __name__ == "__main__":
    main()