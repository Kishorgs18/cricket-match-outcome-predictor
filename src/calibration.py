"""
calibration.py
Probabilistic calibration analysis — the key validation standard for
sports betting pricing models. A well-calibrated model means P=0.70
genuinely wins 70% of the time, which is what separates a pricing
model from a classification model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.calibration import calibration_curve


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE: weighted average miscalibration across confidence bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def plot_calibration_comparison(models_probs: dict, y_true: np.ndarray,
                                 results_dir: str = 'results'):
    """
    Reliability diagram comparing calibration across models.
    Closer to the diagonal = better calibrated = more trustworthy probabilities.
    """
    colors = {'Logistic Regression': 'coral',
              'Random Forest': 'green',
              'Neural Network (ours)': 'steelblue'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Calibration Analysis — Probabilistic Reliability',
                 fontsize=13, fontweight='bold')

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    for name, probs in models_probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10)
        ece = expected_calibration_error(y_true, probs)
        ax.plot(mean_pred, frac_pos, 's-', lw=2,
                color=colors.get(name, 'grey'),
                label=f'{name}  (ECE={ece:.4f})')
    ax.set(xlabel='Mean Predicted Probability', ylabel='Fraction of Positives',
           title='Reliability Diagram')
    ax.legend(fontsize=9)

    # ECE bar chart
    ax2 = axes[1]
    names = list(models_probs.keys())
    eces  = [expected_calibration_error(y_true, p) for p in models_probs.values()]
    bar_colors = [colors.get(n, 'grey') for n in names]
    bars = ax2.bar(names, eces, color=bar_colors, alpha=0.8, edgecolor='black')
    ax2.set(ylabel='Expected Calibration Error (lower = better)',
            title='ECE by Model')
    ax2.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, eces):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'calibration_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return dict(zip(names, eces))


def betting_edge_simulation(y_true: np.ndarray, y_prob: np.ndarray,
                             stake: float = 10.0,
                             results_dir: str = 'results') -> dict:
    """
    Simulate a flat-stake betting strategy using model probabilities as
    implied prices. Bet on Team 1 whenever model probability > 0.5.
    Shows whether the model has a genuine edge over random selection.
    """
    bets = y_prob > 0.5
    wins = bets & (y_true == 1)

    n_bets  = bets.sum()
    n_wins  = wins.sum()
    win_pct = n_wins / n_bets if n_bets > 0 else 0

    # Simple P&L: win stake when correct, lose stake when wrong
    # Assumes implied odds ~ 1 / model_prob (fair odds from model's own price)
    implied_odds = 1.0 / y_prob[bets]
    payouts = np.where(y_true[bets] == 1, stake * (implied_odds - 1), -stake)
    cumulative_pnl = np.cumsum(payouts)

    total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
    roi = total_pnl / (n_bets * stake) * 100

    # Random baseline
    rng = np.random.default_rng(42)
    random_wins = rng.integers(0, 2, size=n_bets)
    random_pnl  = np.sum(np.where(random_wins == 1, stake, -stake))

    print(f"\n{'='*50}")
    print(f"  Betting Edge Simulation")
    print(f"{'='*50}")
    print(f"  Total matches  : {len(y_true)}")
    print(f"  Bets placed    : {n_bets}  (model confidence > 50%)")
    print(f"  Win rate       : {win_pct:.1%}")
    print(f"  Total P&L      : £{total_pnl:+.2f}  (stake £{stake}/bet)")
    print(f"  ROI            : {roi:+.1f}%")
    print(f"  Random baseline P&L: £{random_pnl:+.2f}")
    print(f"{'='*50}")

    # Plot cumulative P&L
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative_pnl, color='steelblue', lw=2, label='Model strategy')
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                    where=(cumulative_pnl >= 0), alpha=0.2, color='green')
    ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                    where=(cumulative_pnl < 0), alpha=0.2, color='red')
    ax.set(xlabel='Bet number', ylabel=f'Cumulative P&L (£)',
           title=f'Flat-stake Betting Simulation  |  ROI: {roi:+.1f}%  |  Win rate: {win_pct:.1%}')
    ax.legend()
    plt.tight_layout()

    path = os.path.join(results_dir, 'betting_edge.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    return {'n_bets': int(n_bets), 'win_rate': win_pct,
            'total_pnl': total_pnl, 'roi': roi}
