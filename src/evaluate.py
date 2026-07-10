"""
evaluate.py
Statistical validation: precision-recall, calibration, confusion matrix,
feature importance via permutation, and results visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import os


def save_fig(fig, name, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def full_evaluation(y_true: np.ndarray, y_prob: np.ndarray, title: str = 'Test Set'):
    y_pred = (y_prob >= 0.5).astype(int)
    auc    = roc_auc_score(y_true, y_prob)
    ap     = average_precision_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)

    print(f"\n{'='*50}")
    print(f"  Evaluation — {title}")
    print(f"{'='*50}")
    print(f"  AUC-ROC        : {auc:.4f}")
    print(f"  Avg Precision  : {ap:.4f}")
    print(f"  Brier Score    : {brier:.4f}  (lower is better; 0 = perfect)")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Team2 Wins', 'Team1 Wins'])}")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle(f'Model Evaluation — {title}', fontsize=14, fontweight='bold')

    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    ax1.legend()

    # Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(rec, prec, color='coral', lw=2, label=f'AP = {ap:.3f}')
    ax2.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    ax2.legend()

    # Calibration Curve
    ax3 = fig.add_subplot(gs[0, 2])
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax3.plot(mean_pred, frac_pos, 's-', color='green', label='Model')
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax3.set(xlabel='Mean Predicted Probability', ylabel='Fraction of Positives',
            title='Calibration Curve')
    ax3.legend()

    # Confusion Matrix
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_true, y_pred)
    im = ax4.imshow(cm, cmap='Blues')
    ax4.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=['Team2 Wins', 'Team1 Wins'],
            yticklabels=['Team2 Wins', 'Team1 Wins'],
            xlabel='Predicted', ylabel='Actual', title='Confusion Matrix')
    for r in range(2):
        for c in range(2):
            ax4.text(c, r, str(cm[r, c]), ha='center', va='center',
                     color='white' if cm[r, c] > cm.max() / 2 else 'black', fontsize=14)

    # Predicted probability distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(y_prob[y_true == 0], bins=30, alpha=0.6, color='coral', label='Team2 wins')
    ax5.hist(y_prob[y_true == 1], bins=30, alpha=0.6, color='steelblue', label='Team1 wins')
    ax5.axvline(0.5, color='black', linestyle='--', lw=1)
    ax5.set(xlabel='Predicted P(Team1 wins)', ylabel='Count',
            title='Predicted Probability Distribution')
    ax5.legend()

    # Training metrics placeholder text
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_text = (
        f"Summary Metrics\n"
        f"{'─'*28}\n"
        f"AUC-ROC       :  {auc:.4f}\n"
        f"Avg Precision :  {ap:.4f}\n"
        f"Brier Score   :  {brier:.4f}\n"
        f"Accuracy      :  {(y_pred == y_true).mean():.4f}\n"
        f"N samples     :  {len(y_true)}"
    )
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax6.axis('off')

    plt.tight_layout()
    save_fig(fig, 'evaluation_report.png')
    return {'auc': auc, 'avg_precision': ap, 'brier': brier}


def plot_training_history(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History', fontsize=13, fontweight='bold')

    axes[0].plot(history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].plot(history['val_loss'],   label='Val Loss',   color='coral')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
    axes[0].legend()

    axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
    axes[1].plot(history['val_auc'], label='Val AUC',      color='purple')
    axes[1].set(xlabel='Epoch', ylabel='Score', title='Validation Metrics')
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, 'training_history.png')


def permutation_feature_importance(model, X_val, y_val, feature_names, device, n_repeats=10):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score
    import torch.nn as nn

    model.eval()
    criterion = nn.CrossEntropyLoss()
    X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_val, dtype=torch.long).to(device)

    with torch.no_grad():
        probs_base = torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()
    base_auc = roc_auc_score(y_val, probs_base)

    importances = []
    for feat_idx in range(X_val.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_val.copy()
            np.random.shuffle(X_perm[:, feat_idx])
            X_pt = torch.tensor(X_perm, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(X_pt), dim=1)[:, 1].cpu().numpy()
            drops.append(base_auc - roc_auc_score(y_val, probs))
        importances.append(np.mean(drops))

    importance_df = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    names  = [x[0] for x in importance_df[:15]]
    values = [x[1] for x in importance_df[:15]]
    colors = ['steelblue' if v > 0 else 'coral' for v in values]
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color='black', lw=0.8)
    ax.set(xlabel='Mean AUC Drop (Permutation Importance)',
           title='Top 15 Feature Importances')
    plt.tight_layout()
    save_fig(fig, 'feature_importance.png')

    print("\nTop 10 Most Important Features:")
    for name, imp in importance_df[:10]:
        print(f"  {name:35s} : {imp:+.4f}")

    return importance_df
