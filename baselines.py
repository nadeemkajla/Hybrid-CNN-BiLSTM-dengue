import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from utils import compute_confusion_metrics
from eval_metrics import select_threshold, evaluate_predictions

def _fit_predict_2d(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    clf.fit(X_train, y_train)
    # decision scores/probabilities
    if hasattr(clf, "predict_proba"):
        p_val  = clf.predict_proba(X_val)[:, 1]
        p_test = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        # map decision_function to probability via logistic link approximation
        from sklearn.preprocessing import MinMaxScaler
        s_val  = clf.decision_function(X_val).reshape(-1, 1)
        s_test = clf.decision_function(X_test).reshape(-1, 1)
        scaler = MinMaxScaler()
        p_val  = scaler.fit_transform(s_val).ravel()
        p_test = scaler.transform(s_test).ravel()
    else:
        # fallback: use predictions as probs
        p_val  = clf.predict(X_val).astype(float)
        p_test = clf.predict(X_test).astype(float)
    thr = select_threshold(y_val, p_val)
    return thr, p_val, p_test

def run_classical_baselines(Xtr3d, ytr, Xva3d, yva, Xte3d, yte):
    # flatten sequences
    Xtr = Xtr3d.reshape(len(Xtr3d), -1)
    Xva = Xva3d.reshape(len(Xva3d), -1)
    Xte = Xte3d.reshape(len(Xte3d), -1)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, n_jobs=-1)
    thr, p_val, p_test = _fit_predict_2d(lr, Xtr, ytr, Xva, yva, Xte, yte)
    results["logreg"] = evaluate_predictions(yte, p_test, thr)

    # Linear SVM
    svm = LinearSVC()
    thr, p_val, p_test = _fit_predict_2d(svm, Xtr, ytr, Xva, yva, Xte, yte)
    results["svm_linear"] = evaluate_predictions(yte, p_test, thr)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400)
    thr, p_val, p_test = _fit_predict_2d(mlp, Xtr, ytr, Xva, yva, Xte, yte)
    results["mlp"] = evaluate_predictions(yte, p_test, thr)

    return results
