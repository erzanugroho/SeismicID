"""Calibration: Platt (sigmoid), Isotonic, Beta. Pick best by Brier on val."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class IdentityCalibrator:
    name = "identity"

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, 1e-6, 1 - 1e-6)


class PlattCalibrator:
    name = "platt"

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    def fit(self, p: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        # Use logit of p as input feature
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        self.lr.fit(x, y)
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self.iso.fit(p, y)
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(p), 1e-6, 1 - 1e-6)


class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): p_calib = sigmoid(a*log(p) + b*log(1-p) + c)."""

    name = "beta"

    def __init__(self) -> None:
        self.lr = LogisticRegression()

    def fit(self, p: np.ndarray, y: np.ndarray) -> BetaCalibrator:
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        feats = np.column_stack([np.log(p), np.log(1 - p)])
        self.lr.fit(feats, y)
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        feats = np.column_stack([np.log(p), np.log(1 - p)])
        return self.lr.predict_proba(feats)[:, 1]


CalibType = PlattCalibrator | IsotonicCalibrator | BetaCalibrator | IdentityCalibrator


def fit_best_calibrator(p_val: np.ndarray, y_val: np.ndarray) -> tuple[CalibType, dict[str, float]]:
    """Try Platt/Isotonic/Beta, pick best by val Brier. Returns (calib, scores_dict)."""
    if len(np.unique(y_val)) < 2:
        # Need both classes for calibration to be meaningful — fall back to identity
        return IdentityCalibrator(), {"identity": brier_score_loss(y_val, p_val)}

    candidates: list[tuple[str, CalibType]] = []
    _fittable: list[PlattCalibrator | IsotonicCalibrator | BetaCalibrator] = [
        PlattCalibrator(), IsotonicCalibrator(), BetaCalibrator(),
    ]
    for cal in _fittable:
        try:
            cal.fit(p_val, y_val)
            candidates.append((cal.name, cal))
        except Exception:  # noqa: BLE001
            continue

    scores: dict[str, float] = {}
    for name, c in candidates:
        try:
            cp = c.predict_proba(p_val)
            scores[name] = float(brier_score_loss(y_val, cp))
        except Exception:  # noqa: BLE001
            continue
    scores["identity"] = float(brier_score_loss(y_val, p_val))

    best_name = min(scores, key=lambda k: scores[k])
    if best_name == "identity":
        return IdentityCalibrator(), scores
    best = next(c for name, c in candidates if name == best_name)
    return best, scores
