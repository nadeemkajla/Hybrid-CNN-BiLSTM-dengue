import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def youdens_j_threshold(y_true, y_prob):
    # returns threshold maximizing TPR - FPR
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    return thr[idx]

def f1_optimal_threshold(y_true, y_prob):
    # grid search thresholds to maximize F1
    from sklearn.metrics import f1_score
    thr_grid = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thr_grid]
    return float(thr_grid[int(np.argmax(scores))])

def compute_confusion_metrics(y_true, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1}

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = inds == b
        if np.any(mask):
            conf = y_prob[mask].mean()
            acc = y_true[mask].mean()
            w = mask.mean()
            ece += w * abs(acc - conf)
    return ece
