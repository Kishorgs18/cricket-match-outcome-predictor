"""
baseline.py
Logistic Regression and Random Forest baselines for model comparison.
Shows the NN is a justified choice over simpler alternatives.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    accuracy_score, roc_curve
)


def run_baselines(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  nn_probs: np.ndarray = None) -> pd.DataFrame:

    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, C=1.0, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=300, max_depth=6,
                                                      random_state=42, n_jobs=-1),
    }

    results = []
    all_probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        all_probs[name] = probs
        results.append({
            'Model':    name,
            'AUC-ROC':  round(roc_auc_score(y_test, probs), 4),
            'Brier':    round(brier_score_loss(y_test, probs), 4),
            'Log Loss': round(log_loss(y_test, probs), 4),
            'Accuracy': round(accuracy_score(y_test, preds), 4),
        })

    if nn_probs is not None:
        nn_preds = (nn_probs >= 0.5).astype(int)
        results.append({
            'Model':    'Neural Network (ours)',
            'AUC-ROC':  round(roc_auc_score(y_test, nn_probs), 4),
            'Brier':    round(brier_score_loss(y_test, nn_probs), 4),
            'Log Loss': round(log_loss(y_test, nn_probs), 4),
            'Accuracy': round(accuracy_score(y_test, nn_preds), 4),
        })
        all_probs['Neural Network (ours)'] = nn_probs

    df = pd.DataFrame(results).set_index('Model')

    print("\n" + "="*60)
    print("  Model Comparison — Hold-out Test Set")
    print("="*60)
    print(df.to_string())
    print("="*60)

    _plot_roc_comparison(all_probs, y_test)
    return df


def _plot_roc_comparison(all_probs: dict, y_test: np.ndarray,
                         results_dir: str = 'results'):
    colors = {'Logistic Regression': 'coral',
              'Random Forest': 'green',
              'Neural Network (ours)': 'steelblue'}

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, 'grey'),
                label=f'{name}  (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve — Model Comparison')
    ax.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'baseline_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
