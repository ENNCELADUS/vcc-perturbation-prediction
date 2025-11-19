import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute MSE and Pearson correlation between true and predicted expression.

    Args:
        y_true: Ground truth expression (samples x genes)
        y_pred: Predicted expression (samples x genes)

    Returns:
        dict: {'mse': float, 'pearson': float}
    """
    # Flatten for overall correlation/MSE
    # Note: In biological context, you might want per-gene or per-cell metrics,
    # but this provides a high-level summary.
    mse = mean_squared_error(y_true, y_pred)

    # Pearson correlation
    # Handle potential constant input (std dev = 0)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        corr = 0.0
    else:
        corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    return {"mse": mse, "pearson": corr}
