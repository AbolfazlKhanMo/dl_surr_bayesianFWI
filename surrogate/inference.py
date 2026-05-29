import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from cnn import CNNRegressor
from train import adaptive

plt.rcParams.update({'font.size': 14})

import logging
logging.basicConfig(
    filename="OUTPUT_FILES/testing.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)


def inverse_standardized_log(arr_scaled, scaler_y):
    return scaler_y.inverse_transform(arr_scaled.reshape(-1, 1))


def log_to_original(arr_log):
    return np.exp(arr_log)


def mc_dropout_predict(model, loader, device, T=100):
    """SOL MC Dropout: only backbone SpatialDropout2d is toggled."""
    all_preds = []
    with torch.no_grad():
        for _ in range(T):
            model.eval()
            model.enable_mc_dropout()
            preds_t = []
            for xb, _ in loader:
                xb = xb.to(device)
                pred = model(xb)
                preds_t.append(pred.cpu().numpy())
            all_preds.append(np.vstack(preds_t))
    model.disable_mc_dropout()
    return np.stack(all_preds, axis=0)   # (T, N, 1)


def calculate_mape(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = predictions.flatten()
    labels = labels.flatten()

    absolute_difference = np.abs(labels - predictions)
    non_zero_mask = (labels != 0)

    if not np.any(non_zero_mask):
        return np.nan

    labels_non_zero = labels[non_zero_mask]
    diff_non_zero = absolute_difference[non_zero_mask]

    ape_non_zero = (diff_non_zero / labels_non_zero) * 100.0
    return np.mean(ape_non_zero)


def compute_metrics(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))

    denom = np.sum((true - true.mean()) ** 2)
    r2 = np.nan if denom == 0 else 1.0 - np.sum((true - pred) ** 2) / denom

    mape = calculate_mape(pred, true)
    return mse, mae, r2, mape


def plot_mc_error_bars_by_index(
    labels_unscaled,
    center_unscaled,
    lower_unscaled,
    upper_unscaled,
    outpath,
):
    y_true = np.asarray(labels_unscaled).reshape(-1)
    y_center = np.asarray(center_unscaled).reshape(-1)
    y_lower = np.asarray(lower_unscaled).reshape(-1)
    y_upper = np.asarray(upper_unscaled).reshape(-1)

    idx = np.arange(len(y_true))

    yerr = np.vstack([
        y_center - y_lower,
        y_upper - y_center
    ])

    plt.figure(figsize=(10, 6))
    plt.plot(idx, y_true, 'k', label="Ground truth")

    plt.errorbar(
        idx,
        y_center,
        yerr=yerr,
        fmt='o',
        markersize=6,
        color='orange',
        ecolor='tab:blue',
        capsize=3,
        elinewidth=2.0,
        alpha=0.9,
        label="MCD mean with 95% interval"
    )

    plt.xlabel("Sample index")
    plt.ylabel("Misfit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()


def plot_mc_error_bars_true_vs_pred(
    labels_unscaled,
    center_unscaled,
    lower_unscaled,
    upper_unscaled,
    outpath,
):
    y_true = np.asarray(labels_unscaled).reshape(-1)
    y_center = np.asarray(center_unscaled).reshape(-1)
    y_lower = np.asarray(lower_unscaled).reshape(-1)
    y_upper = np.asarray(upper_unscaled).reshape(-1)

    idx = np.argsort(y_true)
    y_true = y_true[idx]
    y_center = y_center[idx]
    y_lower = y_lower[idx]
    y_upper = y_upper[idx]

    yerr = np.vstack([
        y_center - y_lower,
        y_upper - y_center
    ])

    plt.figure(figsize=(10, 6))

    dmin = min(y_true.min(), y_center.min())
    dmax = max(y_true.max(), y_center.max())

    plt.plot(
        [dmin, dmax],
        [dmin, dmax],
        'r--',
        linewidth=1.5,
        label="Ideal (y = x)"
    )

    plt.errorbar(
        y_true,
        y_center,
        yerr=yerr,
        fmt='o',
        markersize=6,
        color='orange',
        ecolor='tab:blue',
        capsize=3,
        elinewidth=2.0,
        alpha=0.9,
        label="MCD mean with 95% interval"
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Normalized true misfit")
    plt.ylabel("Normalized Predicted misfit")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()


def plot_mc_relative_uncertainty(labels_unscaled, sigma_log, outpath):
    """Relative uncertainty (%) vs true misfit."""
    y_true = np.asarray(labels_unscaled).reshape(-1)
    s_log = np.asarray(sigma_log).reshape(-1)
    relative_uncertainty = (np.exp(2.0 * s_log) - 1.0) * 100.0
    idx = np.argsort(y_true)
    y_true = y_true[idx]
    relative_uncertainty = relative_uncertainty[idx]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, relative_uncertainty, s=18, color='tab:blue', alpha=0.7)
    plt.xlabel("True misfit")
    plt.ylabel("Relative uncertainty (%)")
    plt.title("Coefficient of variation across misfit range")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()


def plot_residual_band(labels_unscaled, center_unscaled, lower_unscaled, upper_unscaled, outpath):
    y_true = np.asarray(labels_unscaled).reshape(-1)
    y_center = np.asarray(center_unscaled).reshape(-1)
    y_lower = np.asarray(lower_unscaled).reshape(-1)
    y_upper = np.asarray(upper_unscaled).reshape(-1)

    residual = y_center - y_true

    idx = np.argsort(y_true)
    x = y_true[idx]
    residual = residual[idx]
    y_center = y_center[idx]
    y_lower = y_lower[idx]
    y_upper = y_upper[idx]

    lower_res = (y_lower - y_center)[idx]
    upper_res = (y_upper - y_center)[idx]

    plt.figure(figsize=(10, 6))
    plt.scatter(x, residual, s=18, color='orange', alpha=0.8, label="Residual")
    plt.fill_between(
        x,
        lower_res,
        upper_res,
        color='tab:blue',
        alpha=0.25,
        label="Uncertainty interval"
    )
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("True misfit")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=600)
    plt.close()


def test_model():
    scaler_y = joblib.load("OUTPUT_FILES/files/label_scaler.pkl")

    if adaptive:
        data = np.load("OUTPUT_FILES/inference/val.npz")
    else:
        data = np.load("OUTPUT_FILES/inference/test.npz")

    X_test = data["input"]
    y_test = data["label"]

    logging.info(f"Loaded inference data: {X_test.shape}, {y_test.shape}")

    x_t = torch.from_numpy(X_test).float()
    y_t = torch.from_numpy(y_test).float()

    test_loader = DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=64,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRegressor().to(device)
    model.load_state_dict(
        torch.load("OUTPUT_FILES/files/best_model.pt", map_location=device)
    )

    T = 100
    all_preds_scaled = mc_dropout_predict(model, test_loader, device, T=T)

    y_true_log = inverse_standardized_log(y_t.numpy(), scaler_y)
    y_true = log_to_original(y_true_log)

    all_preds_log = np.stack(
        [inverse_standardized_log(all_preds_scaled[t], scaler_y) for t in range(T)],
        axis=0
    )

    mu_log = all_preds_log.mean(axis=0)
    sigma_log = all_preds_log.std(axis=0)

    preds_mc_center = np.exp(mu_log)
    preds_mc_lower = np.exp(mu_log - 2.0 * sigma_log)
    preds_mc_upper = np.exp(mu_log + 2.0 * sigma_log)

    mse_mc, mae_mc, r2_mc, mape_mc = compute_metrics(preds_mc_center, y_true)

    logging.getLogger().handlers[0].stream.write("\n\n")
    logging.info("===== TEST RESULTS (MC DROPOUT MEAN) =====")
    logging.info(f"T   : {T}")
    logging.info(f"MSE : {mse_mc:.6e}")
    logging.info(f"MAE : {mae_mc:.6e}")
    logging.info(f"R^2 : {r2_mc:.4f}")
    logging.info(f"MAPE: {mape_mc:.4f} %")

    plot_mc_error_bars_by_index(
        labels_unscaled=y_true,
        center_unscaled=preds_mc_center,
        lower_unscaled=preds_mc_lower,
        upper_unscaled=preds_mc_upper,
        outpath="OUTPUT_FILES/inference/mc_uncertainty_error_bars_by_index.png"
    )

    plot_mc_error_bars_true_vs_pred(
        labels_unscaled=y_true,
        center_unscaled=preds_mc_center,
        lower_unscaled=preds_mc_lower,
        upper_unscaled=preds_mc_upper,
        outpath="OUTPUT_FILES/inference/mc_uncertainty_error_bars.png"
    )

    plot_mc_relative_uncertainty(
        labels_unscaled=y_true,
        sigma_log=sigma_log,
        outpath="OUTPUT_FILES/inference/mc_relative_uncertainty.png"
    )

    plot_residual_band(
        labels_unscaled=y_true,
        center_unscaled=preds_mc_center,
        lower_unscaled=preds_mc_lower,
        upper_unscaled=preds_mc_upper,
        outpath="OUTPUT_FILES/inference/residual_uncertainty_band.png"
    )


if __name__ == "__main__":
    test_model()