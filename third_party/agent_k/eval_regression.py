# python eval_regression.py --pred pred_y.csv --true test_y.csv --out-json results/metrics.json



#!/usr/bin/env python3
"""
Compute common regression metrics between pred_y.csv and test_y.csv.

Works with CSVs that have:
- Either a single prediction/target column, OR
- (id, target) style columns, where we align rows by id.

Metrics computed:
- MAE, MSE, RMSE, R2
- Median Absolute Error
- MAPE (only if all true y != 0)
- SMAPE (safe around zeros)

Usage examples:
  python eval_regression.py --pred pred_y.csv --true test_y.csv
  python eval_regression.py --pred pred_y.csv --true test_y.csv --id-col id --y-col target
"""

from __future__ import annotations

import json
import os
import argparse
import math
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ID_CANDIDATES = ["id", "ID", "Id"]
DEFAULT_Y_CANDIDATES = ["target", "y", "prediction", "pred", "y_true", "y_pred", "label", "value"]


def _infer_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in DEFAULT_ID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _infer_y_col(df: pd.DataFrame, provided: Optional[str]) -> str:
    if provided:
        if provided not in df.columns:
            raise ValueError(f"Provided y-col '{provided}' not found in columns: {list(df.columns)}")
        return provided

    # Prefer common names
    for c in DEFAULT_Y_CANDIDATES:
        if c in df.columns:
            return c

    # Otherwise, if exactly one non-id column exists, use it
    id_col = _infer_id_col(df)
    candidates = [c for c in df.columns if c != id_col]
    if len(candidates) == 1:
        return candidates[0]

    # Otherwise, if there is exactly one numeric column, use it
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    # Fall back to last column
    return df.columns[-1]


def _load_and_extract(
    path_pred: str,
    path_true: str,
    id_col: Optional[str],
    y_col_pred: Optional[str],
    y_col_true: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    pred_df = pd.read_csv(path_pred)
    true_df = pd.read_csv(path_true)

    # Decide id column
    inferred_pred_id = _infer_id_col(pred_df)
    inferred_true_id = _infer_id_col(true_df)

    if id_col is None:
        # Use id only if both have it
        if inferred_pred_id and inferred_true_id and inferred_pred_id == inferred_true_id:
            id_col = inferred_pred_id
        else:
            id_col = None

    # Decide y columns
    y_pred_col = _infer_y_col(pred_df, y_col_pred)
    y_true_col = _infer_y_col(true_df, y_col_true)

    # Align
    if id_col is not None:
        if id_col not in pred_df.columns:
            raise ValueError(f"Requested id-col '{id_col}' not found in pred CSV columns: {list(pred_df.columns)}")
        if id_col not in true_df.columns:
            raise ValueError(f"Requested id-col '{id_col}' not found in true CSV columns: {list(true_df.columns)}")

        pred_sel = pred_df[[id_col, y_pred_col]].rename(columns={y_pred_col: "y_pred"})
        true_sel = true_df[[id_col, y_true_col]].rename(columns={y_true_col: "y_true"})

        merged = true_sel.merge(pred_sel, on=id_col, how="inner")

        if merged.empty:
            raise ValueError(
                "After merging on id, there were 0 rows. Check that both files share the same ids."
            )

        # Warn if we dropped rows
        if len(merged) != len(true_sel) or len(merged) != len(pred_sel):
            print(
                f"[warn] Merged rows: {len(merged)} | true rows: {len(true_sel)} | pred rows: {len(pred_sel)}",
                file=sys.stderr,
            )

        y_true = merged["y_true"].to_numpy(dtype=float)
        y_pred = merged["y_pred"].to_numpy(dtype=float)
    else:
        # Assume same order / same length
        y_true = true_df[y_true_col].to_numpy(dtype=float)
        y_pred = pred_df[y_pred_col].to_numpy(dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch without id alignment: true has {len(y_true)} rows, pred has {len(y_pred)} rows.\n"
                "Either add an id column to both files or pass --id-col."
            )

    # Drop NaNs pairwise
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"[warn] Dropping {dropped} rows due to NaN/inf in y_true or y_pred.", file=sys.stderr)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        raise ValueError("No valid rows left after filtering NaNs/infs.")

    return y_true, y_pred


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mse(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        # All y_true identical: R2 is ill-defined; follow common convention
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Only valid if y_true != 0 everywhere
    if np.any(y_true == 0):
        return float("nan")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute regression metrics for pred_y.csv vs test_y.csv.")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV (e.g., pred_y.csv).")
    parser.add_argument("--true", required=True, help="Path to ground-truth CSV (e.g., test_y.csv).")
    parser.add_argument("--id-col", default=None, help="ID column name used to align rows (optional).")
    parser.add_argument("--pred-y-col", default=None, help="Prediction column name (optional).")
    parser.add_argument("--true-y-col", default=None, help="Ground-truth column name (optional).")
    parser.add_argument("--out-json", default=None, help="Path to write regression metrics as JSON (e.g., metrics.json).")

    args = parser.parse_args()

    y_true, y_pred = _load_and_extract(
        path_pred=args.pred,
        path_true=args.true,
        id_col=args.id_col,
        y_col_pred=args.pred_y_col,
        y_col_true=args.true_y_col,
    )

    out = {
        "n": int(y_true.size),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MedianAE": median_ae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "SMAPE_%": smape(y_true, y_pred),
    }

    if args.out_json is not None:
        out_path = args.out_json
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)


    # Pretty print
    print("\n=== Regression Metrics ===")
    print(f"n = {out['n']}")
    for k in ["MAE", "MSE", "RMSE", "R2", "MedianAE", "MAPE_%", "SMAPE_%"]:
        v = out[k]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            print(f"{k}: NaN")
        else:
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    print()

    # Also emit one-line CSV-friendly output
    print("csv_line,n,MAE,MSE,RMSE,R2,MedianAE,MAPE_pct,SMAPE_pct")
    print(
        "csv_line,"
        f"{out['n']},"
        f"{out['MAE']},"
        f"{out['MSE']},"
        f"{out['RMSE']},"
        f"{out['R2']},"
        f"{out['MedianAE']},"
        f"{out['MAPE_%']},"
        f"{out['SMAPE_%']}"
    )


if __name__ == "__main__":
    main()
 