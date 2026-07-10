"""
main.py
End-to-end pipeline: data generation -> feature engineering ->
train/test split -> k-fold cross-validation -> final model training ->
full evaluation -> baseline comparison -> calibration analysis ->
betting edge simulation -> feature importance -> save model.
"""

import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import generate_dataset
from src.features import get_feature_matrix, get_scaler
from src.train import train_model, cross_validate
from src.evaluate import full_evaluation, plot_training_history, permutation_feature_importance
from src.baseline import run_baselines
from src.calibration import plot_calibration_comparison, betting_edge_simulation


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs('results', exist_ok=True)

    # 1. Generate / load data
    data_path = 'data/matches.csv'
    if not os.path.exists(data_path):
        print("\nGenerating synthetic match data...")
        import pandas as pd
        df = generate_dataset(n=3000)
    else:
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} matches from {data_path}")

    # 2. Feature engineering
    print("\nEngineering features...")
    X, y, feature_names = get_feature_matrix(df)
    print(f"Feature matrix: {X.shape} | Class balance: {y.mean():.3f}")

    # 3. Train / test split (stratified, 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scale features (fit on train only to prevent leakage)
    scaler = get_scaler(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)

    input_dim = X_train_s.shape[1]
    print(f"Input dimension: {input_dim} features")

    # 5. K-fold cross-validation on training set
    cv_results = cross_validate(
        X_train_s, y_train,
        input_dim=input_dim,
        n_splits=5,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        epochs=80,
        lr=1e-3,
        patience=12,
    )

    # 6. Train final model on full training set
    print("\nTraining final model on full training set...")
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train_s, y_train, test_size=0.15, random_state=0, stratify=y_train
    )

    model, history = train_model(
        X_tr, y_tr, X_vl, y_vl,
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        epochs=100,
        lr=1e-3,
        patience=15,
        device=device,
    )

    # 7. Plot training history
    plot_training_history(history)

    # 8. Full evaluation on held-out test set
    model.eval()
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(X_test_t), dim=1)[:, 1].cpu().numpy()

    metrics = full_evaluation(y_test, probs, title='Hold-out Test Set')

    # 9. Baseline comparison (Logistic Regression + Random Forest vs NN)
    print("\nRunning baseline model comparison...")
    baseline_results = run_baselines(X_train_s, y_train, X_test_s, y_test, nn_probs=probs)

    # 10. Calibration analysis — critical for sports betting pricing validity
    print("\nRunning calibration analysis...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    warnings.filterwarnings('ignore')

    lr  = LogisticRegression(max_iter=5000, C=1.0, random_state=42).fit(X_train_s, y_train)
    rf  = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1).fit(X_train_s, y_train)
    models_probs = {
        'Logistic Regression':    lr.predict_proba(X_test_s)[:, 1],
        'Random Forest':          rf.predict_proba(X_test_s)[:, 1],
        'Neural Network (ours)':  probs,
    }
    ece_scores = plot_calibration_comparison(models_probs, y_test)

    # 11. Betting edge simulation
    print("\nRunning betting edge simulation...")
    edge_results = betting_edge_simulation(y_test, probs)

    # 12. Permutation feature importance
    print("\nComputing permutation feature importance...")
    permutation_feature_importance(model, X_test_s, y_test, feature_names, device)

    # 13. Save model
    model_path = 'results/cricket_outcome_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'feature_names': feature_names,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'test_metrics': metrics,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    print("\n" + "="*50)
    print("  FINAL TEST SET RESULTS")
    print("="*50)
    print(f"  AUC-ROC        : {metrics['auc']:.4f}")
    print(f"  Avg Precision  : {metrics['avg_precision']:.4f}")
    print(f"  Brier Score    : {metrics['brier']:.4f}")
    print(f"  Betting ROI    : {edge_results['roi']:+.1f}%")
    print(f"  Win rate       : {edge_results['win_rate']:.1%}")
    print("="*50)


if __name__ == '__main__':
    main()
