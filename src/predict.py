"""
predict.py
Clean inference module — load the saved model and predict on new match data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union


def load_model(model_path: str = 'results/cricket_outcome_model.pt') -> dict:
    """Load checkpoint. Returns a dict with model, scaler params, and feature names."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    class CricketOutcomeNet(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 2))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = CricketOutcomeNet(
        checkpoint['input_dim'],
        checkpoint['hidden_dims'],
        checkpoint['dropout'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return {
        'model':         model,
        'feature_names': checkpoint['feature_names'],
        'mean':          checkpoint['scaler_mean'],
        'scale':         checkpoint['scaler_scale'],
        'test_metrics':  checkpoint['test_metrics'],
    }


def predict_match(model_data: dict, match_features: dict) -> dict:
    """
    Predict outcome for a single match.

    Parameters
    ----------
    model_data    : output of load_model()
    match_features: dict mapping each of the 29 feature names to a value

    Returns
    -------
    dict with p_team1, p_team2, predicted_winner, confidence
    """
    feature_names = model_data['feature_names']
    missing = [f for f in feature_names if f not in match_features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = np.array([[match_features[f] for f in feature_names]], dtype=np.float32)
    row_scaled = ((row - model_data['mean']) / model_data['scale']).astype(np.float32)

    with torch.no_grad():
        probs = torch.softmax(
            model_data['model'](torch.tensor(row_scaled)), dim=1
        )[0].numpy()

    p1, p2 = float(probs[1]), float(probs[0])
    return {
        'p_team1':          round(p1, 4),
        'p_team2':          round(p2, 4),
        'predicted_winner': 'Team 1' if p1 > 0.5 else 'Team 2',
        'confidence':       round(max(p1, p2), 4),
    }


def predict_batch(model_data: dict, X: np.ndarray) -> np.ndarray:
    """
    Predict on a pre-scaled feature matrix. Returns P(Team 1 wins) for each row.

    Parameters
    ----------
    model_data : output of load_model()
    X          : (n, 29) float32 array — already scaled

    Returns
    -------
    np.ndarray of shape (n,) with win probabilities for Team 1
    """
    X_t = torch.tensor(X.astype(np.float32))
    with torch.no_grad():
        return torch.softmax(model_data['model'](X_t), dim=1)[:, 1].numpy()


if __name__ == '__main__':
    md = load_model()
    print("Model loaded.")
    print(f"Expects {len(md['feature_names'])} features: {md['feature_names']}")
    print(f"Test AUC: {md['test_metrics']['auc']:.4f}")
