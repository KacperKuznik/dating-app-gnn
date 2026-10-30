"""Train GraphSAGE and GAT over several seeds and log every run to MLflow.

Each architecture gets a parent run holding the mean / std of the test metrics,
with one nested run per seed (params, per-epoch curves, test metrics, model).
Browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os
import tempfile

import mlflow
import numpy as np
import torch

import models
import models.gat
from train import COLD_START_RATIO, EPOCHS, LR, SUPERVISION_RATIO, evaluate, train

EXPERIMENT = "speed-dating-match"
ARCHS = {"sage": "GraphSAGE", "gat": "GAT"}
SEEDS = (1, 2, 3, 4, 5)  # the test set is small, so one seed is noisy
METRICS = ["auc", "ap", "precision", "recall", "f1"]


def params(arch):
    p = {
        "arch": arch,
        "hidden_dim": models.HIDDEN_DIM,
        "epochs": EPOCHS,
        "lr": LR,
        "cold_start_ratio": COLD_START_RATIO,
        "supervision_ratio": SUPERVISION_RATIO,
    }
    if arch == "gat":
        p["gat_heads"] = models.gat.HEADS
    return p


def run_seed(arch, seed):
    with mlflow.start_run(run_name=f"{ARCHS[arch]} seed {seed}", nested=True):
        mlflow.log_params({**params(arch), "seed": seed})
        run = train(verbose=False, seed=seed, arch=arch,
                    on_epoch=lambda h: mlflow.log_metrics(
                        {"loss": h["loss"], "val_auc": h["val_auc"], "val_ap": h["val_ap"]}, step=h["epoch"]))
        test = evaluate(run["model"], run["ds"], "test", threshold=run["threshold"])
        mlflow.log_metrics({"best_val_auc": run["best_val_auc"], "threshold": run["threshold"],
                            "test_match_rate": test["match_rate"],
                            **{f"test_{m}": test[m] for m in METRICS}})
        # Same checkpoint format as train.py, so predict.py --model can load it
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"match_{arch}.pt")
            torch.save({"state_dict": run["model"].state_dict(), "threshold": run["threshold"], "arch": arch}, path)
            mlflow.log_artifact(path)
        return test


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    for arch, label in ARCHS.items():
        with mlflow.start_run(run_name=label):
            mlflow.log_params({**params(arch), "seeds": ",".join(map(str, SEEDS))})
            tests = [run_seed(arch, seed) for seed in SEEDS]
            summary = {}
            for m in METRICS:
                values = [t[m] for t in tests]
                summary[f"test_{m}_mean"] = float(np.mean(values))
                summary[f"test_{m}_std"] = float(np.std(values))
            mlflow.log_metrics(summary)
            print(f"{label:10s} test auc {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}   "
                  f"ap {summary['test_ap_mean']:.4f} +/- {summary['test_ap_std']:.4f}   "
                  f"f1 {summary['test_f1_mean']:.4f} +/- {summary['test_f1_std']:.4f}")


if __name__ == "__main__":
    main()
