import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek

from config import (TIMESTAMP_COL, SITE_COL, LABEL_COL, FEATURE_COLS,
                    L_WINDOW, LEAD_H, TRAIN_PROP, VAL_PROP,
                    USE_SMOTE_TOMEK, SMOTE_RANDOM_STATE)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values([TIMESTAMP_COL] if SITE_COL is None else [SITE_COL, TIMESTAMP_COL])
    return df

def _make_windows_site(df_site: pd.DataFrame,
                       feature_cols: List[str],
                       L: int,
                       h: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    arr = df_site[feature_cols].values
    y = df_site[LABEL_COL].values
    n = len(df_site)
    # indices where a full window ending at t exists and label at t+h exists
    for t in range(L - 1, n - h):
        X_list.append(arr[t - L + 1: t + 1, :])
        y_list.append(y[t + h])
    if not X_list:
        return np.empty((0, L, len(feature_cols))), np.empty((0,))
    return np.stack(X_list, axis=0), np.asarray(y_list)

def make_windows(df: pd.DataFrame,
                 feature_cols: List[str],
                 L: int,
                 h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns X (N,L,d), y (N,), t_end timestamps (for splitting)
    X_all, y_all, t_end_all = [], [], []
    if SITE_COL is None:
        X, y = _make_windows_site(df, feature_cols, L, h)
        X_all.append(X); y_all.append(y)
        t_end_all.append(df[TIMESTAMP_COL].values[L-1:len(df)-h])
    else:
        for _, g in df.groupby(SITE_COL):
            X, y = _make_windows_site(g, feature_cols, L, h)
            if len(X):
                X_all.append(X); y_all.append(y)
                t_end_all.append(g[TIMESTAMP_COL].values[L-1:len(g)-h])
    X = np.concatenate(X_all, axis=0) if X_all else np.empty((0, L, len(feature_cols)))
    y = np.concatenate(y_all, axis=0) if y_all else np.empty((0,))
    t_end = np.concatenate(t_end_all, axis=0) if t_end_all else np.empty((0,))
    return X, y, t_end

def blocked_time_split(t_end: np.ndarray, train_prop: float, val_prop: float):
    # Split by quantiles of end timestamps
    assert train_prop + val_prop < 1.0
    n = len(t_end)
    idx_sorted = np.argsort(t_end)
    q1 = int(np.floor(n * train_prop))
    q2 = int(np.floor(n * (train_prop + val_prop)))
    train_idx = idx_sorted[:q1]
    val_idx = idx_sorted[q1:q2]
    test_idx = idx_sorted[q2:]
    return train_idx, val_idx, test_idx

def scale_train_only(X_train, X_val, X_test):
    L, d = X_train.shape[1], X_train.shape[2]
    scaler = StandardScaler().fit(X_train.reshape(len(X_train), -1))
    Xtr = scaler.transform(X_train.reshape(len(X_train), -1)).reshape(-1, L, d)
    Xva = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(-1, L, d)
    Xte = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1, L, d)
    return Xtr, Xva, Xte, scaler

def smote_tomek_train_only(X_train, y_train):
    X2d = X_train.reshape(len(X_train), -1)
    smt = SMOTETomek(random_state=SMOTE_RANDOM_STATE)
    Xb, yb = smt.fit_resample(X2d, y_train)
    L, d = X_train.shape[1], X_train.shape[2]
    Xb = Xb.reshape(-1, L, d)
    return Xb, yb

def prepare_dataset(csv_path: str,
                    feature_cols: Optional[List[str]] = None,
                    L: int = L_WINDOW,
                    h: int = LEAD_H):
    df = load_csv(csv_path)
    feats = feature_cols or FEATURE_COLS
    X, y, t_end = make_windows(df, feats, L, h)
    train_idx, val_idx, test_idx = blocked_time_split(t_end, TRAIN_PROP, VAL_PROP)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    X_train, X_val, X_test, scaler = scale_train_only(X_train, X_val, X_test)

    if USE_SMOTE_TOMEK:
        X_train, y_train = smote_tomek_train_only(X_train, y_train)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feats
