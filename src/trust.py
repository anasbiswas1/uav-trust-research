"""Trust-layer metrics and calibration utilities for UAV security detectors.

Model-agnostic, pure functions over NumPy arrays of predicted probabilities and
integer labels. Reused across datasets (network IDS and navigation attacks).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def logit(p, eps=1e-7):
    """Numerically safe logit of a probability array."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def top_label_ece(probs, labels, n_bins=15):
    """Top-label Expected Calibration Error (equal-width bins).

    probs: shape (N,) as P(positive) or (N, K) of class probabilities.
    labels: integer class labels.
    """
    probs = np.asarray(probs, float)
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    conf = probs.max(1); preds = probs.argmax(1); labels = np.asarray(labels)
    acc = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); n = len(labels); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def brier_binary(p_pos, labels):
    """Brier score for binary predictions: P(positive) against {0, 1} labels."""
    return float(np.mean((np.asarray(p_pos, float) - np.asarray(labels, float)) ** 2))


def conformal_qhat(cal_probs, cal_labels, alpha=0.1):
    """Split-conformal threshold (LAC / score method) from a calibration set."""
    cp = np.asarray(cal_probs, float)
    if cp.ndim == 1:
        cp = np.column_stack([1 - cp, cp])
    n = len(cal_labels)
    scores = 1 - cp[np.arange(n), np.asarray(cal_labels)]
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def coverage_size(probs, labels, qhat):
    """Empirical coverage and mean prediction-set size for a conformal threshold."""
    p = np.asarray(probs, float)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    sets = p >= (1 - qhat); labels = np.asarray(labels)
    return float(sets[np.arange(len(labels)), labels].mean()), float(sets.sum(1).mean())


def aurc(conf, correct):
    """Area under the risk-coverage curve (lower is better).

    Returns (aurc, coverages, selective_risk).
    """
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    order = np.argsort(-conf); cs = correct[order]; k = np.arange(1, len(cs) + 1)
    selective_risk = 1 - np.cumsum(cs) / k
    return float(selective_risk.mean()), k / len(cs), selective_risk


def bootstrap_ci(fn, *arrays, B=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for metric fn(*arrays). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed); n = len(arrays[0]); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(fn(*[a[idx] for a in arrays]))
    out = np.array(out)
    return (float(out.mean()),
            float(np.percentile(out, 100 * alpha / 2)),
            float(np.percentile(out, 100 * (1 - alpha / 2))))


def fit_temperature(cal_logits, cal_labels):
    """Fit a single temperature by NLL on binary calibration logits."""
    lg = np.asarray(cal_logits, float); y = np.asarray(cal_labels, float)
    def nll(T):
        p = np.clip(1 / (1 + np.exp(-lg / T)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def fit_calibrators(cal_logits, cal_probs, cal_labels):
    """Fit temperature, Platt, and isotonic calibrators on the calibration set."""
    return {
        "temperature": fit_temperature(cal_logits, cal_labels),
        "platt": LogisticRegression().fit(np.asarray(cal_logits).reshape(-1, 1), cal_labels),
        "isotonic": IsotonicRegression(out_of_bounds="clip").fit(cal_probs, cal_labels),
    }


def apply_calibrators(fitted, test_logits, test_probs):
    """Apply fitted calibrators to a test set. Returns dict name -> P(positive)."""
    lg = np.asarray(test_logits, float); p = np.asarray(test_probs, float)
    return {
        "raw": p,
        "temperature": 1 / (1 + np.exp(-lg / fitted["temperature"])),
        "platt": fitted["platt"].predict_proba(lg.reshape(-1, 1))[:, 1],
        "isotonic": fitted["isotonic"].transform(p),
    }
