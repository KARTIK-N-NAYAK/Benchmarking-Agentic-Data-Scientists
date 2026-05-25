#!/usr/bin/env python3

# python eval_classification.py --pred pred_y.csv --true test_y.csv --out-json results/metrics.json --proba

from __future__ import annotations
import argparse, json, os
from typing import Optional
import numpy as np
import pandas as pd

DEFAULT_ID_CANDIDATES = ["id", "ID", "Id"]
DEFAULT_Y_CANDIDATES = ["target", "y", "prediction", "pred", "y_true", "y_pred", "label", "class", "category", "value"]


# ----------------------------
# Helpers
# ----------------------------

def _infer_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in DEFAULT_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _infer_y_col(df: pd.DataFrame, provided: Optional[str]) -> str:
    if provided:
        if provided not in df.columns:
            raise ValueError(f"{provided} not in {list(df.columns)}")
        return provided

    for c in DEFAULT_Y_CANDIDATES:
        if c in df.columns:
            return c

    id_col = _infer_id_col(df)
    candidates = [c for c in df.columns if c != id_col]

    if len(candidates) == 1:
        return candidates[0]

    return df.columns[-1]


def _is_probability_format(df: pd.DataFrame, id_col: Optional[str]) -> bool:
    cols = [c for c in df.columns if c != id_col]

    # multiclass probabilities
    if len(cols) > 1 and all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
        return True

    # binary probability (single column between 0 and 1)
    if len(cols) == 1:
        col = cols[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna().values
            if len(values) > 0 and np.all((values >= 0) & (values <= 1)):
                return True

    return False


# ----------------------------
# Loading
# ----------------------------

def _load_align_true_pred_labels(path_pred, path_true, id_col, y_col_pred, y_col_true):
    pred_df = pd.read_csv(path_pred)
    true_df = pd.read_csv(path_true)

    # Handle one-hot or probability format in true_df
    if y_col_true is None and _is_probability_format(true_df, id_col):
        cols_true = [c for c in true_df.columns if c != id_col]
        if len(cols_true) > 1:
            y_true_series = true_df[cols_true].idxmax(axis=1)
            if id_col:
                true_sel = pd.DataFrame({id_col: true_df[id_col], "y_true": y_true_series})
            else:
                y_true = y_true_series.to_numpy()
        else:
            y_true_col = cols_true[0]
            if id_col:
                true_sel = true_df[[id_col, y_true_col]].rename(columns={y_true_col: "y_true"})
            else:
                y_true = true_df[y_true_col].to_numpy()
    else:
        y_true_col = _infer_y_col(true_df, y_col_true)
        if id_col:
            true_sel = true_df[[id_col, y_true_col]].rename(columns={y_true_col: "y_true"})
        else:
            y_true = true_df[y_true_col].to_numpy()

    # Handle one-hot or probability format in pred_df
    if y_col_pred is None and _is_probability_format(pred_df, id_col):
        cols_pred = [c for c in pred_df.columns if c != id_col]
        if len(cols_pred) > 1:
            y_pred_series = pred_df[cols_pred].idxmax(axis=1)
            if id_col:
                pred_sel = pd.DataFrame({id_col: pred_df[id_col], "y_pred": y_pred_series})
            else:
                y_pred = y_pred_series.to_numpy()
        else:
            y_pred_col = cols_pred[0]
            if id_col:
                pred_sel = pred_df[[id_col, y_pred_col]].rename(columns={y_pred_col: "y_pred"})
            else:
                y_pred = pred_df[y_pred_col].to_numpy()
    else:
        y_pred_col = _infer_y_col(pred_df, y_col_pred)
        if id_col:
            pred_sel = pred_df[[id_col, y_pred_col]].rename(columns={y_pred_col: "y_pred"})
        else:
            y_pred = pred_df[y_pred_col].to_numpy()

    if id_col:
        merged = true_sel.merge(pred_sel, on=id_col, how="inner")
        if merged.empty:
            raise ValueError("Merge resulted in 0 rows — ID mismatch")
        y_true = merged["y_true"].to_numpy()
        y_pred = merged["y_pred"].to_numpy()

    return y_true, y_pred


def _load_align_true_and_proba(path_pred, path_true, id_col, y_col_true):
    pred_df = pd.read_csv(path_pred)
    true_df = pd.read_csv(path_true)

    proba_cols = [c for c in pred_df.columns if c != id_col]

    # Handle one-hot or probability format in true_df
    if y_col_true is None and _is_probability_format(true_df, id_col):
        cols_true = [c for c in true_df.columns if c != id_col]
        if len(cols_true) > 1:
            # Align labels to the same class order as pred_df if possible
            common_cols = [c for c in proba_cols if c in cols_true]
            if len(common_cols) > 1:
                y_true_series = true_df[common_cols].idxmax(axis=1)
            else:
                y_true_series = true_df[cols_true].idxmax(axis=1)

            if id_col:
                true_sel = pd.DataFrame(
                    {id_col: true_df[id_col], "y_true": y_true_series}
                )
            else:
                y_true = y_true_series.to_numpy()
        else:
            y_true_col = cols_true[0]
            if id_col:
                true_sel = true_df[[id_col, y_true_col]].rename(columns={y_true_col: "y_true"})
            else:
                y_true = true_df[y_true_col].to_numpy()
    else:
        y_true_col = _infer_y_col(true_df, y_col_true)
        if id_col:
            true_sel = true_df[[id_col, y_true_col]].rename(columns={y_true_col: "y_true"})
        else:
            y_true = true_df[y_true_col].to_numpy()

    if id_col:
        pred_sel = pred_df[[id_col] + proba_cols]
        merged = true_sel.merge(pred_sel, on=id_col, how="inner")

        if merged.empty:
            raise ValueError("Merge resulted in 0 rows — ID mismatch")

        y_true = merged["y_true"].to_numpy()
        proba = merged[proba_cols].to_numpy(dtype=float)
    else:
        proba = pred_df[proba_cols].to_numpy(dtype=float)

    return y_true, proba, proba_cols


# ----------------------------
# Metrics
# ----------------------------

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, classes):
    if len(classes) > 1000:
        raise ValueError(
            f"Detected {len(classes)} classes. Likely probabilities treated as labels."
        )

    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[idx[str(t)], idx[str(p)]] += 1

    return cm


def per_class_precision_recall_f1(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    return precision, recall, f1, support


def balanced_accuracy_from_cm(cm):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return float(np.mean(recall))


def log_loss(y_true, proba, class_names, eps=1e-15):
    y_true = np.array(y_true).astype(str)

    # --- FIX: binary single-column case ---
    if proba.shape[1] == 1:
        p1 = np.clip(proba[:, 0], eps, 1 - eps)
        p0 = 1 - p1
        proba = np.vstack([p0, p1]).T
        # Use provided class_names (should have 2 elements)

    idx = {str(c): i for i, c in enumerate(class_names)}
    y_idx = np.array([idx[str(t)] for t in y_true])

    p = np.clip(proba, eps, 1 - eps)
    p /= p.sum(axis=1, keepdims=True)

    return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx])))


def roc_auc_score_multiclass(y_true, proba, class_names):
    from sklearn.metrics import roc_auc_score

    y_true = np.array(y_true).astype(str)

    idx = {str(c): i for i, c in enumerate(class_names)}
    y_idx = np.array([idx[str(t)] for t in y_true])

    # --- FIX: binary single-column case ---
    if proba.shape[1] == 1:
        # P(y=class_names[1])
        return float(roc_auc_score(y_idx, proba[:, 0]))

    if len(class_names) == 2:
        return float(roc_auc_score(y_idx, proba[:, 1]))
    else:
        # Pass labels to handle cases where some classes are missing in y_true
        return float(
            roc_auc_score(
                y_idx, proba, multi_class="ovr", labels=np.arange(len(class_names))
            )
        )


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--true", required=True)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--pred-y-col", default=None)
    parser.add_argument("--true-y-col", default=None)
    parser.add_argument("--proba", action="store_true")
    parser.add_argument("--out-json", default=None)

    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred)
    id_col = args.id_col or _infer_id_col(pred_df)

    is_proba = args.proba or _is_probability_format(pred_df, id_col)

    if is_proba:
        y_true, proba, class_names = _load_align_true_and_proba(
            args.pred, args.true, id_col, args.true_y_col
        )

        # ---- FIX: handle binary vs multiclass properly ----
        if proba.shape[1] == 1:
            # Single column probability: usually P(y=1)
            # Try to infer class names from y_true if they are not 0/1 (e.g., Class_1, Class_2)
            unique_true = sorted(list(set(y_true.astype(str))))
            if len(unique_true) == 2:
                class_names = unique_true
            else:
                class_names = ["0", "1"]
            
            y_pred = np.array([class_names[1] if p > 0.5 else class_names[0] for p in proba[:, 0]])
        else:
            y_pred = np.array([str(class_names[i]) for i in np.argmax(proba, axis=1)])
            class_names = [str(c) for c in class_names]

    else:
        y_true, y_pred = _load_align_true_pred_labels(
            args.pred, args.true, id_col, args.pred_y_col, args.true_y_col
        )
        class_names = sorted(list(set(y_true.astype(str)) | set(y_pred.astype(str))))
        proba = np.zeros((len(y_pred), len(class_names)))
        for i, c in enumerate(class_names):
            proba[:, i] = (y_pred.astype(str) == c).astype(float)

    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    # Reconcile y_true with class_names if there's a mismatch (e.g. '1' vs 'Class_1')
    if is_proba:
        missing = set(y_true) - set(class_names)
        if missing:
            # If all missing labels are numeric strings, try prefixing with 'Class_'
            if all(m.isdigit() for m in missing):
                mapped_y_true = np.array([f"Class_{t}" if t.isdigit() and f"Class_{t}" in class_names else t for t in y_true])
                if set(mapped_y_true).issubset(set(class_names)):
                    y_true = mapped_y_true
            # Or if class_names are numeric and y_true is 'Class_X'
            elif all(t.startswith("Class_") for t in missing):
                mapped_y_true = np.array([t.replace("Class_", "") if t.startswith("Class_") and t.replace("Class_", "") in class_names else t for t in y_true])
                if set(mapped_y_true).issubset(set(class_names)):
                    y_true = mapped_y_true

    cm = confusion_matrix(y_true, y_pred, class_names)
    prec, rec, f1, support = per_class_precision_recall_f1(cm)

    acc = accuracy(y_true, y_pred)
    bal_acc = balanced_accuracy_from_cm(cm)

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * support) / np.sum(support))
    micro_f1 = acc

    out = {
        "Accuracy": acc,
        "BalancedAccuracy": bal_acc,
        "F1_macro": macro_f1,
        "F1_weighted": weighted_f1,
        "F1_micro": micro_f1,
    }

    if proba is not None:
        out["LogLoss"] = log_loss(y_true, proba, class_names)
        out["ROC_AUC"] = roc_auc_score_multiclass(y_true, proba, class_names)

    print("\n=== Metrics ===")
    for k, v in out.items():
        print(f"{k}: {v:.6f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()