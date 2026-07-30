import numpy as np
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import normalize
from sklearn.svm import SVR


def _fit_predict(X_train, Y_train, X_eval, C=1.0):
    model = MultiOutputRegressor(SVR(kernel="rbf", C=C))
    model.fit(X_train, Y_train)
    preds = model.predict(X_eval)
    return normalize(preds, norm="l2", axis=1)


def kfold_residual_metrics(X, Y, k=5):
    """5-fold CV SVR; summarizes L2 residual norms between prediction and target."""
    kf = KFold(n_splits=k)
    y_pred = np.zeros_like(Y)

    for train_idx, test_idx in kf.split(X):
        y_pred[test_idx] = _fit_predict(X[train_idx], Y[train_idx], X[test_idx])

    residual_norms = np.linalg.norm(Y - y_pred, axis=1)
    metrics = {
        "Mean_Residual_Norm": np.mean(residual_norms),
        "Median_Residual_Norm": np.median(residual_norms),
        "90th_Percentile_Residual_Norm": np.percentile(residual_norms, 90),
    }
    return {key: round(float(val), 6) for key, val in metrics.items()}


def conformal_metrics(X_train, Y_train, X_cal, Y_cal, X_test, Y_test, alpha=0.10):
    
    model = MultiOutputRegressor(SVR(kernel="rbf", C=1.0))
    model.fit(X_train, Y_train)

    cal_preds = normalize(model.predict(X_cal), norm="l2", axis=1)
    # r_i = || e_target_i - f(e_source_i) ||_2
    cal_residuals = np.linalg.norm(Y_cal - cal_preds, axis=1)

    n = len(cal_residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_threshold = np.quantile(cal_residuals, q_level, method="higher")

    test_preds = normalize(model.predict(X_test), norm="l2", axis=1)
    test_residuals = np.linalg.norm(Y_test - test_preds, axis=1)
    # C(x) = { e : || e - f(x) ||_2 <= q_threshold }
    is_covered = test_residuals <= q_threshold

    empirical_coverage = np.mean(is_covered)
    metrics = {
        "empirical_coverage": empirical_coverage,
        "mean_radius": q_threshold,
        "coverage_error": abs(empirical_coverage - (1 - alpha)),
    }
    return {key: round(float(val), 6) for key, val in metrics.items()}
