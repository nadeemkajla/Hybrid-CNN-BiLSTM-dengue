import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score)

from utils import compute_confusion_metrics, expected_calibration_error, youdens_j_threshold, f1_optimal_threshold
from config import THRESHOLD_STRATEGY, NUM_ECE_BINS

def select_threshold(y_val, p_val):
    if THRESHOLD_STRATEGY.lower() == "youden":
        return youdens_j_threshold(y_val, p_val)
    return f1_optimal_threshold(y_val, p_val)

def evaluate_predictions(y_true, p_prob, threshold):
    y_hat = (p_prob >= threshold).astype(int)
    metrics = compute_confusion_metrics(y_true, y_hat)
    metrics["auroc"] = roc_auc_score(y_true, p_prob)
    metrics["auprc"] = average_precision_score(y_true, p_prob)
    metrics["brier"] = brier_score_loss(y_true, p_prob)
    metrics["ece"]   = expected_calibration_error(y_true, p_prob, n_bins=NUM_ECE_BINS)
    metrics["threshold"] = float(threshold)
    return metrics
