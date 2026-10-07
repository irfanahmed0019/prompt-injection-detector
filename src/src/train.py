"""Compare detectors with 5-fold CV on train, evaluate the chosen one once on the held-out test split."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import load
from .models import candidates

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def scores(y, pred, s):
    return {"precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, s)}


def score_of(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def bootstrap_ci(y, pred, n=1000):
    rng = np.random.default_rng(SEED)
    y, pred = np.asarray(y), np.asarray(pred)
    f = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        f.append(f1_score(y[i], pred[i]))
    return [float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))]


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    report = {"dataset": "deepset/prompt-injections", "n_train": len(Xtr), "n_test": len(Xte), "cv": {}}
    for name, pipe in candidates().items():
        r = cross_validate(clone(pipe), Xtr, ytr, cv=cv, scoring=["f1", "precision", "recall"])
        report["cv"][name] = {k: round(float(r[f"test_{k}"].mean()), 3) for k in ["f1", "precision", "recall"]}
        print(name, report["cv"][name])
    best = max((n for n in report["cv"] if n != "majority_baseline"), key=lambda n: report["cv"][n]["f1"])
    model = clone(candidates()[best]).fit(Xtr, ytr)
    pred = model.predict(Xte)
    m = {k: round(float(v), 3) for k, v in scores(yte, pred, score_of(model, Xte)).items()}
    m["f1_95ci"] = [round(x, 3) for x in bootstrap_ci(yte, pred)]
    base = candidates()["majority_baseline"].fit(Xtr, ytr).predict(Xte)
    m["majority_baseline_f1"] = round(float(f1_score(yte, base)), 3)
    report["selected"], report["test"] = best, m
    misses = [Xte[i][:140] for i in range(len(yte)) if yte[i] == 1 and pred[i] == 0]
    report["missed_injection_examples"] = misses[:5]
    report["false_positive_examples"] = [Xte[i][:140] for i in range(len(yte)) if yte[i] == 0 and pred[i] == 1][:5]
    print("selected", best, m)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "detector.joblib")


if __name__ == "__main__":
    main()
