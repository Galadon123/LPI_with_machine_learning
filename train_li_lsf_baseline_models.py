import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


G = 9.81  # m/s^2


def compute_rd_simplified(depth_m: float) -> float:
    """
    Matches the current project sheet approximation in build_merged_spt_from_spt_value.py:
      rd = 1 - 0.015 * z
    """
    if depth_m is None or np.isnan(depth_m):
        return np.nan
    z = float(depth_m)
    rd = 1.0 - 0.015 * z
    return rd


def compute_csr7p5_from_a_stresses(a_max_g: float, sigma_v_kpa: float, sigma_v_eff_kpa: float, depth_m: float) -> float:
    """
    Matches build_merged_spt_from_spt_value.py compute_csr:
      CSR = 0.65 * a_max_g * (sigma_v / sigma_v') * rd
    """
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in [a_max_g, sigma_v_kpa, sigma_v_eff_kpa, depth_m]):
        return np.nan
    if sigma_v_eff_kpa <= 0:
        return np.nan
    rd = compute_rd_simplified(depth_m)
    if rd is None or np.isnan(rd):
        return np.nan
    if rd <= 0:
        return np.nan
    return 0.65 * float(a_max_g) * (float(sigma_v_kpa) / float(sigma_v_eff_kpa)) * rd


def compute_cn(sigma_v_eff_kpa: float) -> float:
    """
    Same as build_merged_spt_from_spt_value.py:
      CN = 0.77 * log10(2000 / sigma_v_eff)
    """
    if sigma_v_eff_kpa is None or np.isnan(sigma_v_eff_kpa) or float(sigma_v_eff_kpa) <= 0:
        return np.nan
    ratio = 2000.0 / float(sigma_v_eff_kpa)
    if ratio <= 0:
        return np.nan
    cn = 0.77 * math.log10(ratio)
    if cn <= 0 or np.isnan(cn):
        return np.nan
    return cn


def grnn_kernel_weights(X_train: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    # weights ~ exp(-||x-xi||^2 / (2*sigma^2))
    # X_train and x are already scaled.
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    diff = X_train - x  # (n, d)
    d2 = np.einsum("ij,ij->i", diff, diff)  # squared euclidean distances
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    return w


@dataclass
class GRNNClassifier:
    sigma: float
    X_train: np.ndarray
    y_train: np.ndarray  # in {0,1}

    def predict_proba_one(self, x: np.ndarray) -> float:
        w = grnn_kernel_weights(self.X_train, x, self.sigma)
        denom = float(np.sum(w))
        if denom <= 0:
            return 0.5
        p = float(np.sum(w * self.y_train) / denom)
        # Numerical safety
        return float(np.clip(p, 1e-9, 1.0 - 1e-9))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # vectorized over rows (still O(n*m) but manageable for <= ~2k)
        preds = np.zeros((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            preds[i] = self.predict_proba_one(X[i])
        # return probability of class 1
        return preds


@dataclass
class GRNNRegressor:
    sigma: float
    X_train: np.ndarray
    y_train: np.ndarray  # continuous

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            w = grnn_kernel_weights(self.X_train, X[i], self.sigma)
            denom = float(np.sum(w))
            if denom <= 0:
                preds[i] = float(np.mean(self.y_train))
            else:
                preds[i] = float(np.sum(w * self.y_train) / denom)
        return preds


class LIWrapper:
    """
    Pickle-friendly wrapper for LI models.
    chosen_obj formats:
      - {"kind": "single", "model": model}
      - {"kind": "ensemble", "top2_keys": [k0,k1], "models": {k0:model0, k1:model1}}
    """

    def __init__(self, scaler: StandardScaler, chosen_obj: Dict):
        self.scaler = scaler
        self.chosen_obj = chosen_obj

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_raw)
        if self.chosen_obj["kind"] == "single":
            m = self.chosen_obj["model"]
            if isinstance(m, GRNNClassifier):
                return m.predict_proba(Xs)
            return m.predict_proba(Xs)[:, 1]

        k0, k1 = self.chosen_obj["top2_keys"]
        m0 = self.chosen_obj["models"][k0]
        m1 = self.chosen_obj["models"][k1]
        p0 = m0.predict_proba(Xs) if isinstance(m0, GRNNClassifier) else m0.predict_proba(Xs)[:, 1]
        p1 = m1.predict_proba(Xs) if isinstance(m1, GRNNClassifier) else m1.predict_proba(Xs)[:, 1]
        return np.clip(0.5 * (p0 + p1), 1e-9, 1 - 1e-9)


class LSFWrapper:
    """
    Pickle-friendly wrapper for LSF regression models.
    chosen_obj formats:
      - {"kind": "single", "model": model}
      - {"kind": "ensemble", "models": [model0, model1]}
    """

    def __init__(self, scaler: StandardScaler, chosen_obj: Dict):
        self.scaler = scaler
        self.chosen_obj = chosen_obj

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_raw)
        if self.chosen_obj["kind"] == "ensemble":
            m0, m1 = self.chosen_obj["models"]
            return 0.5 * (m0.predict(Xs) + m1.predict(Xs))
        return self.chosen_obj["model"].predict(Xs)


def build_historical_feature_table(
    hanna_csv: Path,
    hwang_csv: Path,
    *,
    use_spt_only: bool = True,
) -> pd.DataFrame:
    """
    Harmonize Hanna (Table A1) and Hwang (Appendix I) into a common SPT-based feature table.

    LI inputs (common):
      - N1_60 (computed for Hwang from N60 and CN)
      - FCI (use fines content proxy)
      - sigma_v_eff_kpa (effective vertical stress)
      - CSR7.5 (computed from amax, stresses, rd)
    Label:
      - T=1 for liquefaction occurrence (Yes / Liq* / etc)
      - T=0 for non-liquefaction
    """
    hanna = pd.read_csv(hanna_csv)
    hwang = pd.read_csv(hwang_csv)

    # ---- Hanna ----
    # columns:
    # Z_m, N1_60, Fp75mm_pct, dw_m, svo_kPa, s0vo_kPa, at_g, tav_over_s0vo, Vs_mps, f0_deg, Mv, amax_g, Liquefaction
    hanna_feat = pd.DataFrame(
        {
            "N1_60": hanna["N1_60"].astype(float),
            # fines proxy
            "FCI": hanna["Fp75mm_pct"].astype(float),
            "sigma_v_eff_kpa": hanna["s0vo_kPa"].astype(float),
            "CSR7.5": [
                compute_csr7p5_from_a_stresses(a, svo, s0, z)
                for a, svo, s0, z in zip(hanna["amax_g"], hanna["svo_kPa"], hanna["s0vo_kPa"], hanna["Z_m"])
            ],
            "T": (hanna["Liquefaction"].astype(str).str.lower().str.strip() == "yes").astype(int),
            "Source": "Hanna2007",
        }
    )

    # ---- Hwang ----
    # columns:
    # Liquefied (0/1 derived), N60, Depth_m, FC_pct, sigma_v_prime_t_m2, sigma_v_t_m2, PGA_g, Mw_or_extra
    # label:
    #   Liquefied is already available from the earlier extraction (1 for Liq*, 0 for nonLiq*)
    # features:
    #   N1_60 computed from N60 and CN based on sigma_v_eff
    #   sigma_v_eff converted from t/m^2 to kPa
    t_m2_to_kpa = 9.80665
    hwang_sigv_eff = hwang["sigma_v_prime_t_m2"].astype(float) * t_m2_to_kpa
    hwang_sigv = hwang["sigma_v_t_m2"].astype(float) * t_m2_to_kpa

    hwang_cn = [compute_cn(seff) for seff in hwang_sigv_eff]
    hwang_n1_60 = hwang["N60"].astype(float).values * np.array(hwang_cn, dtype=float)

    # PGA_g is already amax/g
    hwang_csr = [
        compute_csr7p5_from_a_stresses(pga_g, sv, s0, z)
        for pga_g, sv, s0, z in zip(hwang["PGA_g"], hwang_sigv, hwang_sigv_eff, hwang["Depth_m"])
    ]

    hwang_feat = pd.DataFrame(
        {
            "N1_60": hwang_n1_60,
            "FCI": hwang["FC_pct"].astype(float),
            "sigma_v_eff_kpa": hwang_sigv_eff.values,
            "CSR7.5": hwang_csr,
            "T": hwang["Liquefied"].astype(int),
            "Source": "Hwang2021",
        }
    )

    df = pd.concat([hanna_feat, hwang_feat], ignore_index=True)

    # Basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["N1_60", "FCI", "sigma_v_eff_kpa", "CSR7.5", "T"])
    df["T"] = df["T"].astype(int)

    return df


def evaluate_li_model(name: str, y_true: np.ndarray, p1: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    out: Dict[str, float] = {}
    out["model"] = name
    try:
        out["AUROC"] = float(roc_auc_score(y_true, p1))
    except Exception:
        out["AUROC"] = float("nan")
    out["LogLoss"] = float(log_loss(y_true, np.vstack([1 - p1, p1]).T))
    out["Brier"] = float(brier_score_loss(y_true, p1))
    return out


def train_and_select_li_models(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    grnn_sigmas: Optional[List[float]] = None,
) -> Tuple[object, Dict[str, float], List[Dict[str, float]]]:
    """
    Train baseline LI models and pick best single model or ensemble on validation.

    LI features used:
      X = [N1_60, FCI, sigma_v_eff_kpa, CSR7.5]
    """
    if grnn_sigmas is None:
        grnn_sigmas = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]

    feats = ["N1_60", "FCI", "sigma_v_eff_kpa", "CSR7.5"]
    X = df[feats].values.astype(float)
    y = df["T"].values.astype(int)

    # Split: train / val / test
    # First split off test, then val from remaining.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss1.split(X, y))
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    # val from train_full
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=random_state)
    train_idx2, val_idx2 = next(sss2.split(X_train_full, y_train_full))
    X_train, X_val = X_train_full[train_idx2], X_train_full[val_idx2]
    y_train, y_val = y_train_full[train_idx2], y_train_full[val_idx2]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ---- Candidate 1: GRNN classifier ----
    grnn_results = []
    best_grnn = None
    best_grnn_metrics = None
    for sigma in grnn_sigmas:
        grnn = GRNNClassifier(sigma=sigma, X_train=X_train_s, y_train=y_train)
        p1_val = grnn.predict_proba(X_val_s)
        metrics = evaluate_li_model(f"GRNN(sigma={sigma})", y_val, p1_val)
        grnn_results.append(metrics)
        if best_grnn_metrics is None or (metrics["LogLoss"] < best_grnn_metrics["LogLoss"]):
            best_grnn = grnn
            best_grnn_metrics = metrics

    # ---- Candidate 2: MLP classifier ----
    mlp_candidates = [
        (8,),
        (16,),
        (32,),
        (8, 8),
        (16, 8),
    ]
    mlp_alphas = [1e-4, 5e-4, 1e-3]
    mlp_lrs = [1e-3, 5e-4]

    best_mlp = None
    best_mlp_metrics = None
    for hl in mlp_candidates:
        for alpha in mlp_alphas:
            for lr in mlp_lrs:
                clf = MLPClassifier(
                    hidden_layer_sizes=hl,
                    activation="tanh",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=lr,
                    max_iter=2000,
                    random_state=random_state,
                )
                clf.fit(X_train_s, y_train)
                p1_val = clf.predict_proba(X_val_s)[:, 1]
                metrics = evaluate_li_model(f"MLP{hl}(alpha={alpha},lr={lr})", y_val, p1_val)
                if best_mlp_metrics is None or (metrics["LogLoss"] < best_mlp_metrics["LogLoss"]):
                    best_mlp = clf
                    best_mlp_metrics = metrics

    # ---- Candidate 3: SVM classifier ----
    # Note: SVC probability=True includes internal Platt scaling.
    svm_Cs = [0.5, 1, 2, 5, 10]
    svm_gammas = ["scale", 0.05, 0.1, 0.2, 0.5]
    best_svm = None
    best_svm_metrics = None
    for C in svm_Cs:
        for gamma in svm_gammas:
            clf = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=random_state,
            )
            clf.fit(X_train_s, y_train)
            p1_val = clf.predict_proba(X_val_s)[:, 1]
            metrics = evaluate_li_model(f"SVM(C={C},gamma={gamma})", y_val, p1_val)
            if best_svm_metrics is None or (metrics["LogLoss"] < best_svm_metrics["LogLoss"]):
                best_svm = clf
                best_svm_metrics = metrics

    # Rank candidates by LogLoss (lower is better)
    candidate_list = [best_grnn_metrics, best_mlp_metrics, best_svm_metrics]
    candidate_objs = {"grnn": best_grnn, "mlp": best_mlp, "svm": best_svm}

    # Create a simple best-single winner by validation logloss
    all_candidates = []
    if best_grnn_metrics:
        all_candidates.append(("grnn", best_grnn, best_grnn_metrics))
    if best_mlp_metrics:
        all_candidates.append(("mlp", best_mlp, best_mlp_metrics))
    if best_svm_metrics:
        all_candidates.append(("svm", best_svm, best_svm_metrics))

    all_candidates.sort(key=lambda t: (t[2]["LogLoss"], -t[2]["AUROC"]))
    best_single_key, best_single_obj, best_single_metrics = all_candidates[0][0], all_candidates[0][1], all_candidates[0][2]

    # Ensemble: average probabilities from top-2 by val LogLoss
    top2 = all_candidates[:2]
    def ensemble_predict_proba_s(Xs: np.ndarray) -> np.ndarray:
        ps = []
        for _, obj, _ in top2:
            if hasattr(obj, "predict_proba"):
                # sklearn-style classifier has predict_proba returning (n,2)
                # GRNNClassifier.predict_proba returns p1 already.
                if isinstance(obj, GRNNClassifier):
                    ps.append(obj.predict_proba(Xs))
                else:
                    ps.append(obj.predict_proba(Xs)[:, 1])
            else:
                raise RuntimeError("Unknown model type")
        p_avg = np.mean(np.vstack(ps), axis=0)
        return np.clip(p_avg, 1e-9, 1 - 1e-9)

    p1_val_ens = ensemble_predict_proba_s(X_val_s)
    ens_metrics = evaluate_li_model(f"Ensemble({top2[0][0]}+{top2[1][0]})", y_val, p1_val_ens)

    # Choose between best single and ensemble based on validation LogLoss
    if ens_metrics["LogLoss"] < best_single_metrics["LogLoss"]:
        chosen = {
            "kind": "ensemble",
            "top2_keys": [top2[0][0], top2[1][0]],
            "models": {top2[0][0]: top2[0][1], top2[1][0]: top2[1][1]},
        }
        chosen_metrics_val = ens_metrics
        chosen_name = chosen["kind"]
    else:
        chosen = {"kind": "single", "key": best_single_key, "model": best_single_obj}
        chosen_metrics_val = best_single_metrics
        chosen_name = f"single({best_single_key})"

    li_wrapper = LIWrapper(scaler=scaler, chosen_obj=chosen)

    # Final evaluation on test (report only)
    X_test_s = scaler.transform(X_test)
    if chosen["kind"] == "single":
        m = chosen["model"]
        p1_test = m.predict_proba(X_test_s) if isinstance(m, GRNNClassifier) else m.predict_proba(X_test_s)[:, 1]
    else:
        k0, k1 = chosen["top2_keys"]
        m0 = chosen["models"][k0]
        m1 = chosen["models"][k1]
        p0 = m0.predict_proba(X_test_s) if isinstance(m0, GRNNClassifier) else m0.predict_proba(X_test_s)[:, 1]
        p1 = m1.predict_proba(X_test_s) if isinstance(m1, GRNNClassifier) else m1.predict_proba(X_test_s)[:, 1]
        p1_test = np.clip((p0 + p1) / 2.0, 1e-9, 1 - 1e-9)
    test_metrics = evaluate_li_model(f"chosen={chosen_name}", y_test, p1_test)

    results_rows = []
    results_rows.extend(all_candidates)

    # Save LI artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "li_wrapper.pkl", "wb") as f:
        pickle.dump(li_wrapper, f)
    # Also save a compact summary
    summary = {
        "chosen_name": chosen_name,
        "val": {k: float(chosen_metrics_val[k]) for k in ["LogLoss", "AUROC", "Brier"] if k in chosen_metrics_val},
        "test": {k: float(test_metrics[k]) for k in ["LogLoss", "AUROC", "Brier"] if k in test_metrics},
        "top_candidates": [
            {
                "key": k,
                "LogLoss": float(m["LogLoss"]),
                "AUROC": float(m["AUROC"]),
                "Brier": float(m["Brier"]),
            }
            for k, _, m in all_candidates
        ],
        "all_grnn_sigmas": grnn_results,
    }
    (output_dir / "li_selection_summary.json").write_text(json.dumps(summary, indent=2))

    # Return chosen wrapper and a quick chosen test metrics
    return li_wrapper, test_metrics, [all_candidates[0][2], all_candidates[1][2], all_candidates[2][2]]


def boundary_search_generate_crr_targets(
    df: pd.DataFrame,
    li_model: object,
    *,
    csr_col: str = "CSR7.5",
    p05: float = 0.5,
    max_bisect_iter: int = 30,
    max_bracket_iter: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each record i:
      - Fix soil indices: (N1_60, FCI, sigma_v_eff_kpa)
      - Vary CSR7.5 input to LI model until P(T=1) == 0.5 (approx)
      - Set CRR_target = CSR_crit (since limit state boundary implies critical CSR=CRR)
    """
    soil_cols = ["N1_60", "FCI", "sigma_v_eff_kpa"]
    x_cols = soil_cols + [csr_col]

    X_base = df[x_cols].values.astype(float)
    y_label = df["T"].values.astype(int)

    csr_base = df[csr_col].values.astype(float)
    out_crr = np.full((len(df),), np.nan, dtype=float)

    # Bisection in CSR scale (log space tends to behave better)
    for i in range(len(df)):
        s = X_base[i, :3]  # soil indices
        csr0 = csr_base[i]
        if np.isnan(csr0) or csr0 <= 0:
            continue

        # helper to predict probability
        def p_at(csr_val: float) -> float:
            row = np.array([s[0], s[1], s[2], csr_val], dtype=float).reshape(1, -1)
            p = float(li_model.predict_proba(row)[0])
            return float(p)

        # Initial bracket depending on label direction
        # We want p(low) < 0.5 < p(high)
        if y_label[i] == 1:
            high = csr0
            low = csr0 * 0.01
            # Ensure p(low) < 0.5
            for _ in range(max_bracket_iter):
                try:
                    pl = p_at(low)
                except Exception:
                    pl = np.nan
                if pl < p05:
                    break
                low *= 0.5
                if low <= 1e-12:
                    break
            # Ensure p(high) >= 0.5
            for _ in range(max_bracket_iter):
                ph = p_at(high)
                if ph >= p05:
                    break
                high *= 2.0
                if high >= csr0 * 1e6:
                    break
        else:
            low = csr0
            high = csr0 * 2.0
            # Ensure p(low) < 0.5
            for _ in range(max_bracket_iter):
                pl = p_at(low)
                if pl < p05:
                    break
                low *= 0.5
                if low <= 1e-12:
                    break
            # Ensure p(high) >= 0.5
            for _ in range(max_bracket_iter):
                ph = p_at(high)
                if ph >= p05:
                    break
                high *= 2.0
                if high >= csr0 * 1e6:
                    break

        try:
            pl = p_at(low)
            ph = p_at(high)
        except Exception:
            continue
        if not (pl < p05 and ph >= p05):
            continue

        # Bisection in log CSR
        log_low = math.log(low)
        log_high = math.log(high)
        best_csr = None
        best_diff = 1e9
        for _ in range(max_bisect_iter):
            log_mid = 0.5 * (log_low + log_high)
            csr_mid = math.exp(log_mid)
            pm = p_at(csr_mid)
            diff = abs(pm - p05)
            if diff < best_diff:
                best_diff = diff
                best_csr = csr_mid
            if pm < p05:
                log_low = log_mid
            else:
                log_high = log_mid
        out_crr[i] = float(best_csr) if best_csr is not None else np.nan

    out = df.copy()
    out["CRR_target"] = out_crr
    return out


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan")}
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae}


def train_and_select_lsf_models(
    df_with_targets: pd.DataFrame,
    *,
    output_dir: Path,
    random_state: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    grnn_sigmas: Optional[List[float]] = None,
) -> Tuple[object, Dict[str, float], Dict[str, float]]:
    """
    Train CRR regressors to predict CRR_target from soil indices:
      X = [N1_60, FCI, sigma_v_eff_kpa]
      y = CRR_target
    """
    if grnn_sigmas is None:
        grnn_sigmas = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]

    soil_cols = ["N1_60", "FCI", "sigma_v_eff_kpa"]
    feats = soil_cols
    df = df_with_targets.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["CRR_target"])

    X = df[feats].values.astype(float)
    y = df["CRR_target"].values.astype(float)

    # For regression splits, just stratify by label if available to keep similar distribution
    if "T" in df.columns:
        strat_y = df["T"].values.astype(int)
    else:
        strat_y = np.zeros((len(df),), dtype=int)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss1.split(X, strat_y))
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
    )
    train_idx2, val_idx2 = next(sss2.split(X_train_full, strat_y[train_idx]))
    X_train, X_val = X_train_full[train_idx2], X_train_full[val_idx2]
    y_train, y_val = y_train_full[train_idx2], y_train_full[val_idx2]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ---- GRNN regressor ----
    best_grnn = None
    best_grnn_rmse = float("inf")
    for sigma in grnn_sigmas:
        grnn = GRNNRegressor(sigma=sigma, X_train=X_train_s, y_train=y_train)
        ypred_val = grnn.predict(X_val_s)
        metrics = evaluate_regression(y_val, ypred_val)
        if metrics["RMSE"] < best_grnn_rmse:
            best_grnn = grnn
            best_grnn_rmse = metrics["RMSE"]
            best_grnn_metrics = metrics

    # ---- MLP regressor ----
    mlp_candidates = [
        (8,),
        (16,),
        (32,),
        (16, 8),
        (32, 16),
    ]
    mlp_alphas = [1e-5, 1e-4, 5e-4]
    mlp_lrs = [1e-3, 5e-4]
    best_mlp = None
    best_mlp_metrics = None
    best_mlp_rmse = float("inf")
    for hl in mlp_candidates:
        for alpha in mlp_alphas:
            for lr in mlp_lrs:
                reg = MLPRegressor(
                    hidden_layer_sizes=hl,
                    activation="tanh",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=lr,
                    max_iter=4000,
                    random_state=random_state,
                )
                reg.fit(X_train_s, y_train)
                ypred_val = reg.predict(X_val_s)
                metrics = evaluate_regression(y_val, ypred_val)
                if metrics["RMSE"] < best_mlp_rmse:
                    best_mlp = reg
                    best_mlp_metrics = metrics
                    best_mlp_rmse = metrics["RMSE"]

    # ---- SVR regressor ----
    svr_Cs = [1, 5, 10, 50]
    svr_gammas = ["scale", 0.05, 0.1, 0.2]
    svr_eps = [0.01, 0.05, 0.1]
    best_svr = None
    best_svr_metrics = None
    best_svr_rmse = float("inf")
    for C in svr_Cs:
        for gamma in svr_gammas:
            for eps in svr_eps:
                reg = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=eps)
                reg.fit(X_train_s, y_train)
                ypred_val = reg.predict(X_val_s)
                metrics = evaluate_regression(y_val, ypred_val)
                if metrics["RMSE"] < best_svr_rmse:
                    best_svr = reg
                    best_svr_metrics = metrics
                    best_svr_rmse = metrics["RMSE"]

    # Rank by val RMSE
    candidates = [
        ("grnn", best_grnn, best_grnn_metrics),
        ("mlp", best_mlp, best_mlp_metrics),
        ("svr", best_svr, best_svr_metrics),
    ]
    candidates.sort(key=lambda t: (t[2]["RMSE"], t[2]["MAE"]))

    best1 = candidates[0]
    best2 = candidates[1]

    # Ensemble (avg predictions of top2) on val
    def pred_ensemble(Xs: np.ndarray) -> np.ndarray:
        p1 = best1[1].predict(Xs)
        p2 = best2[1].predict(Xs)
        return 0.5 * (p1 + p2)

    ypred_val_ens = pred_ensemble(X_val_s)
    ens_metrics = evaluate_regression(y_val, ypred_val_ens)

    if ens_metrics["RMSE"] <= best1[2]["RMSE"]:
        chosen = {"kind": "ensemble", "models": [best1[1], best2[1]]}
        chosen_name = f"Ensemble({best1[0]}+{best2[0]})"
        chosen_val_metrics = ens_metrics
    else:
        chosen = {"kind": "single", "model": best1[1]}
        chosen_name = f"single({best1[0]})"
        chosen_val_metrics = best1[2]

    # Test evaluation
    if chosen["kind"] == "ensemble":
        ypred_test = pred_ensemble(X_test_s)
    else:
        ypred_test = chosen["model"].predict(X_test_s)
    test_metrics = evaluate_regression(y_test, ypred_test)

    lsf_wrapper = LSFWrapper(scaler=scaler, chosen_obj=chosen)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lsf_wrapper.pkl", "wb") as f:
        pickle.dump(lsf_wrapper, f)

    summary = {
        "chosen_name": chosen_name,
        "val": {k: float(chosen_val_metrics[k]) for k in ["RMSE", "MAE"]},
        "test": {k: float(test_metrics[k]) for k in ["RMSE", "MAE"]},
        "candidates": [
            {"key": k, "RMSE": float(m["RMSE"]), "MAE": float(m["MAE"])}
            for k, _, m in candidates
        ],
        "ensemble_val": {"RMSE": float(ens_metrics["RMSE"]), "MAE": float(ens_metrics["MAE"])},
    }
    (output_dir / "lsf_selection_summary.json").write_text(json.dumps(summary, indent=2))

    # Return wrapper and summaries
    return lsf_wrapper, test_metrics, chosen_val_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanna_csv", default="hanna2007_historical_cases.csv")
    ap.add_argument("--hwang_csv", default="hwang2021_historical_cases.csv")
    ap.add_argument("--out_dir", default="historical_ann_training_outputs")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_historical_feature_table(Path(args.hanna_csv), Path(args.hwang_csv))
    (out_dir / "combined_historical_features.csv").write_text(df.to_csv(index=False))

    # Train/select LI model
    li_dir = out_dir / "LI"
    li_model, li_test_metrics, _ = train_and_select_li_models(df, output_dir=li_dir, random_state=args.random_state)

    # Generate CRR targets using the chosen LI model
    # We'll do this for all rows, then LSF training will re-split.
    crd_df = boundary_search_generate_crr_targets(df, li_model)
    (out_dir / "boundary_search_crr_targets.csv").write_text(crd_df.to_csv(index=False))

    # Train/select LSF model
    lsf_dir = out_dir / "LSF"
    lsf_wrapper, lsf_test_metrics, lsf_val_metrics = train_and_select_lsf_models(
        crd_df,
        output_dir=lsf_dir,
        random_state=args.random_state,
    )

    # Top-level summary
    summary = {
        "li_test": li_test_metrics,
        "lsf_val": lsf_val_metrics,
        "lsf_test": lsf_test_metrics,
        "n_rows_combined": int(len(df)),
        "n_rows_with_crr_target": int(crd_df["CRR_target"].notna().sum()),
        "li_artifact": str(li_dir / "li_wrapper.pkl"),
        "lsf_artifact": str(lsf_dir / "lsf_wrapper.pkl"),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))

    print("Training complete.")
    print("Outputs:", out_dir)


if __name__ == "__main__":
    main()

