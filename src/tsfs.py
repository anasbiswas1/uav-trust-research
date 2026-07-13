"""Transfer-Stable Feature Selection (TSFS).

Selects features that generalize across attack families using ONLY the seen
families. A feature is transfer-fragile if its SHAP attribution reverses from
pro-attack to pro-normal on an unseen family; TSFS estimates this without the
unseen family, via an internal leave-one-seen-family-out loop, and drops the
most fragile features before training the final detector. This turns the
fragility diagnosis into a practical, leakage-free defence against
attack-family distribution shift.
"""
import numpy as np
import shap


def mean_shap(explainer, X):
    """Mean per-feature SHAP value on a sample (binary; positive -> attack class)."""
    s = np.asarray(explainer.shap_values(X))
    if s.ndim == 3:
        s = s[..., 1]
    return s.mean(0)


def reversal_vector(m_reference, m_target):
    """Per-feature attribution reversal: pro-attack on reference, pro-normal on target."""
    return np.maximum(np.asarray(m_reference), 0.0) * np.maximum(-np.asarray(m_target), 0.0)


def internal_fragility(fit_model, X, y, fam, normal_value, seen_families,
                       n_features, n_shap=2000, seed=0):
    """Estimate per-feature transfer-fragility using only the seen families.

    fit_model(X_tr, y_tr, seed) must return a fitted tree classifier with
    predict_proba. For each seen family f, a model is trained on normal traffic
    plus the OTHER seen families, and the reversal of each feature's attribution
    on f (a held-in proxy for an unseen family) is accumulated. The unseen test
    family never enters this computation.
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
    return acc / max(len(seen_families), 1)


def select_stable_features(fragility, drop_fraction=0.2):
    """Return indices of the transfer-stable features (drop the most-fragile fraction)."""
    fragility = np.asarray(fragility)
    p = len(fragility)
    n_drop = int(round(drop_fraction * p))
    if n_drop <= 0:
        return np.arange(p)
    drop = set(np.argsort(-fragility)[:n_drop].tolist())
    return np.array([j for j in range(p) if j not in drop])
