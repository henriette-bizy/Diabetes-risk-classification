"""
Evaluation, experiment tracking and plotting helpers.

Every experiment in the project (scikit-learn and Keras) goes through the same
`ExperimentTracker`, so all runs are scored with identical metrics, written to one
CSV table, and logged to Weights & Biases in offline mode. The W&B run folders in
`logs/wandb/` are the evidence that backs the experiment table in the report.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import label_binarize

SHORT_NAMES = ["no_diab", "prediab", "diab"]


def compute_metrics(y_true, y_proba):
    """All metrics used in the project, computed from predicted probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.argmax(y_proba, axis=1)
    labels = list(range(y_proba.shape[1]))
    y_bin = label_binarize(y_true, classes=labels)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc_macro": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
        "pr_auc_macro": average_precision_score(y_bin, y_proba, average="macro"),
    }
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, name in enumerate(SHORT_NAMES[: len(labels)]):
        m[f"recall_{name}"] = rec[i]
        m[f"precision_{name}"] = prec[i]
        m[f"f1_{name}"] = f1[i]
    return {k: float(v) for k, v in m.items()}


class ExperimentTracker:
    """Collects every run into one table and mirrors it to W&B (offline)."""

    def __init__(self, log_path, project="diabetes-risk-classification",
                 use_wandb=True, wandb_dir="logs"):
        self.log_path = log_path
        self.project = project
        self.use_wandb = use_wandb
        self.wandb_dir = wandb_dir
        self.records = []
        if use_wandb:
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_SILENT", "true")
            os.makedirs(wandb_dir, exist_ok=True)

    def log(self, exp_id, approach, model_name, params, features, imbalance,
            y_true, y_proba, train_time=None, train_macro_f1=None, notes="",
            extra=None):
        metrics = compute_metrics(y_true, y_proba)
        record = {
            "exp_id": exp_id, "approach": approach, "model": model_name,
            "features": features, "imbalance": imbalance, "eval_split": "validation",
            "params": json.dumps(params, default=str),
            "train_time_s": None if train_time is None else round(train_time, 1),
            "train_macro_f1": train_macro_f1, **metrics, "notes": notes,
        }
        if extra:
            record.update(extra)
        # Re-running a cell replaces that experiment instead of duplicating it
        self.records = [r for r in self.records if r["exp_id"] != exp_id] + [record]
        pd.DataFrame(self.records).to_csv(self.log_path, index=False)
        if self.use_wandb:
            self._log_wandb(record, params, metrics)
        return metrics

    def _log_wandb(self, record, params, metrics):
        try:
            import wandb
            run = wandb.init(
                project=self.project, name=f"{record['exp_id']}_{record['model']}",
                dir=self.wandb_dir, mode="offline",
                config={"exp_id": record["exp_id"], "approach": record["approach"],
                        "model": record["model"], "features": record["features"],
                        "imbalance": record["imbalance"], **params},
            )
            summary = {f"val/{k}": v for k, v in metrics.items()}
            if record["train_macro_f1"] is not None:
                summary["train/macro_f1"] = record["train_macro_f1"]
            if record["train_time_s"] is not None:
                summary["train_time_s"] = record["train_time_s"]
            run.log(summary)
            run.finish()
        except Exception as err:  # logging should never stop an experiment
            print(f"W&B logging skipped: {err}")

    def table(self, columns=None):
        df = pd.DataFrame(self.records)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        return df


def fit_and_log(tracker, exp_id, model, X_train, y_train, X_val, y_val, *,
                model_name, params, features, imbalance, notes="", fit_kwargs=None,
                train_eval_size=30_000, seed=42):
    """Fit a scikit-learn style model, time it, and log validation metrics.

    Training macro-F1 is measured on a random sample of the training set so the
    train/validation gap (a quick overfitting check) is cheap to compute.
    """
    start = time.time()
    model.fit(X_train, y_train, **(fit_kwargs or {}))
    train_time = time.time() - start

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_train), size=min(train_eval_size, len(X_train)), replace=False)
    y_tr = np.asarray(y_train)[idx]
    train_f1 = f1_score(y_tr, model.predict(X_train[idx]), average="macro", zero_division=0)

    y_proba = model.predict_proba(X_val)
    metrics = tracker.log(exp_id, "Traditional ML", model_name, params, features, imbalance,
                          y_val, y_proba, train_time=train_time,
                          train_macro_f1=float(train_f1), notes=notes)
    print(f"{exp_id} {model_name}: val macro-F1 {metrics['macro_f1']:.4f} | "
          f"train macro-F1 {train_f1:.4f} | ROC-AUC {metrics['roc_auc_macro']:.4f} | "
          f"recall prediab {metrics['recall_prediab']:.3f}, diab {metrics['recall_diab']:.3f} | "
          f"{train_time:.1f}s")
    return model, metrics


class FittedClassifier:
    """Bundles the fitted preprocessing, optional column selection and model,
    so the saved file can take raw survey answers and return predictions."""

    def __init__(self, preprocessor, model, column_idx=None, class_names=None):
        self.preprocessor = preprocessor
        self.model = model
        self.column_idx = column_idx
        self.class_names = class_names

    def _transform(self, X_raw):
        X = self.preprocessor.transform(X_raw)
        return X if self.column_idx is None else X[:, self.column_idx]

    def predict_proba(self, X_raw):
        return self.model.predict_proba(self._transform(X_raw))

    def predict(self, X_raw):
        return np.argmax(self.predict_proba(X_raw), axis=1)


def plot_confusion_matrices(y_true, y_pred, class_names, title, save_path=None):
    """Raw counts next to row-normalised recall, since counts alone hide how
    badly the small classes are handled."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[1],
                vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Counts")
    axes[1].set_title("Row-normalised (recall per true class)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    return cm


def plot_roc_curves(y_true, y_proba, class_names, title, save_path=None):
    """One-vs-rest ROC curve for each class."""
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
