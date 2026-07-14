"""Transfer-Stable Feature Selection (TSFS).

Selects features that generalize across attack families using ONLY the seen
families, via an internal leave-one-seen-family-out estimate of attribution
reversal. Memory-managed: each internal model and SHAP explainer is released
immediately, since the nested loop otherwise accumulates native memory.
"""
import gc
import numpy as np
import shap


def mean_shap(explainer, X):
    s = np.asarray(explainer.shap_values(X))
    if s.ndim == 3:
        s = s[..., 1]
    return s.mean(0)


def reversal_vector(m_reference, m_target):
    """Per-feature reversal: pro-attack on reference, pro-normal on target."""
    return np.maximum(np.asarray(m_reference), 0.0) * np.maximum(-np.asarray(m_target), 0.0)


def internal_fragility(fit_model, X, y, fam, normal_value, seen_families,
                       n_features, n_shap=1000, seed=0):
    """Estimate per-feature transfer-fragility using only the seen families.

    fit_model(X_tr, y_tr, seed) returns a fitted tree classifier with predict_proba.
    Each fold's model and explainer are deleted before the next to bound memory.
    """
    fam = np.asarray(fam)
    acc = np.zeros(n_features)
    rng = np.random.default_rng(seed)
    for i, f in enumerate(seen_families):
        others = [g for g in seen_families if g != f]
        tr = (fam == normal_value) | np.isin(fam, others)
        clf = fit_model(X[tr], y[tr], seed + i)
        expl = shap.TreeExplainer(clf)

        def samp(mask):
            idx = np.where(mask)[0]
            if len(idx) > n_shap:
                idx = rng.choice(idx, n_shap, replace=False)
            return X[idx]

        m_ref = mean_shap(expl, samp(np.isin(fam, others)))
        m_f = mean_shap(expl, samp(fam == f))
        acc += reversal_vector(m_ref, m_f)
        del clf, expl
        gc.collect()
    return acc / max(len(seen_families), 1)


def select_stable_features(fragility, drop_fraction=0.2):
    """Indices of transfer-stable features (drop the most-fragile fraction)."""
    fragility = np.asarray(fragility)
    p = len(fragility)
    n_drop = int(round(drop_fraction * p))
    if n_drop <= 0:
        return np.arange(p)
    drop = set(np.argsort(-fragility)[:n_drop].tolist())
    return np.array([j for j in range(p) if j not in drop])
