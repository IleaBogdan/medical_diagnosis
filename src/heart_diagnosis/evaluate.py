"""Evaluate the multi-agent system on the UCI Heart Disease dataset."""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .crew import HeartDiagnosisCrew
from .data_loader import load_dataset, patient_to_dict


def evaluate(
    engine: str = "auto",
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 42,
    model: Optional[str] = None,
    llm_coordinator: bool = False,
    protocol: str = "single",
    strategy: str = "weighted",
    use_critic: bool = True,
    use_risk: bool = True,
    output_path: str = "evaluation_report.json",
    verbose: bool = True,
) -> dict:
    df = load_dataset()
    if sample is not None:
        df = df.sample(n=min(sample, len(df)), random_state=seed).reset_index(drop=True)
    elif limit is not None:
        df = df.head(limit).reset_index(drop=True)

    crew = HeartDiagnosisCrew(
        engine=engine, model=model, llm_coordinator=llm_coordinator,
        protocol=protocol, strategy=strategy, use_critic=use_critic, use_risk=use_risk,
    )

    y_true, y_pred, y_score = [], [], []
    rows = []
    t0 = time.time()

    for i, row in df.iterrows():
        patient = patient_to_dict(row)
        result = crew.diagnose(patient)
        pred = 1 if result.final.final_decision == "Positive" else 0
        true = int(row["target"])

        y_true.append(true)
        y_pred.append(pred)
        y_score.append(result.final.final_score)
        rows.append(
            {
                "index": int(i),
                "true": true,
                "pred": pred,
                "final_score": result.final.final_score,
                "agreement_level": result.final.agreement_level,
                "correct": pred == true,
            }
        )
        if verbose:
            mark = "OK " if pred == true else "XX "
            print(
                f"[{i + 1}/{len(df)}] {mark} true={true} pred={pred} "
                f"score={result.final.final_score:.2f} "
                f"agree={result.final.agreement_level:.2f}"
            )

    elapsed = time.time() - t0

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    except ValueError:
        metrics["roc_auc"] = None

    report = {
        "engine": crew.engine,
        "protocol": crew.protocol,
        "strategy": crew.strategy,
        "use_critic": crew.use_critic,
        "use_risk": crew.use_risk,
        "n_patients": len(df),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "agents": [
            {"name": a.name, "role": a.role.strip(), "features": a.features,
             "reliability": a.reliability}
            for a in crew.agents
        ],
        "per_patient": rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if verbose:
        print("\n==================== EVALUATION SUMMARY ====================")
        print(f"Engine     : {crew.engine}")
        print(f"Protocol   : {crew.protocol}   Strategy: {crew.strategy}   "
              f"critic={crew.use_critic} risk={crew.use_risk}")
        print(f"Patients   : {len(df)}")
        print(f"Accuracy   : {metrics['accuracy']}")
        print(f"Precision  : {metrics['precision']}")
        print(f"Recall     : {metrics['recall']}")
        print(f"F1 score   : {metrics['f1']}")
        print(f"ROC AUC    : {metrics['roc_auc']}")
        print(f"Confusion  : {metrics['confusion_matrix']}  (rows=true, cols=pred)")
        print(f"Report     : {output_path}")
        print("===========================================================")

    return report
