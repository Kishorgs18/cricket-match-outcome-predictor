# Cricket Match Outcome Predictor

A PyTorch feedforward neural network that predicts T20 cricket match outcomes using in-play and pre-match statistical features.

## Overview

The model ingests 29 engineered features covering team form, batting and bowling metrics, powerplay dynamics, wickets in hand, and pitch conditions to predict the probability of each team winning a T20 match.

**Results on held-out test set (600 matches)**

| Metric | Score |
|---|---|
| AUC-ROC | 0.7469 |
| Average Precision | 0.8686 |
| Brier Score | 0.1746 |
| CV Accuracy (5-fold) | 0.763 ± 0.024 |
| CV AUC (5-fold) | 0.791 ± 0.032 |

## Project Structure

```
cricket-match-outcome-predictor/
├── data/
│   ├── generate_data.py     # Synthetic T20 match data generator (3000 matches)
│   └── matches.csv          # Generated dataset
├── src/
│   ├── model.py             # CricketOutcomeNet architecture
│   ├── features.py          # Feature engineering pipeline
│   ├── train.py             # Training loop, early stopping, k-fold CV
│   └── evaluate.py          # Evaluation, visualisation, permutation importance
├── results/
│   ├── training_history.png
│   ├── evaluation_report.png
│   └── feature_importance.png
├── main.py                  # End-to-end pipeline entry point
└── requirements.txt
```

## Model Architecture

```
Input (29 features)
   |
Linear(29 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
   |
Linear(128 -> 64)  -> BatchNorm1d -> ReLU -> Dropout(0.3)
   |
Linear(64 -> 32)   -> BatchNorm1d -> ReLU -> Dropout(0.3)
   |
Linear(32 -> 2)    -> Softmax
```

Weight initialisation: Kaiming normal (He init) for all Linear layers.

## Features

**Pre-match features (15)**
- Team form win percentage (last 10 matches)
- Head to head win ratio
- Batting average, bowling economy
- Venue win percentage
- Toss winner and decision
- Pitch type (batting | balanced | bowling, encoded as 1 | 0 | -1)

**In-play features (6 raw + 8 engineered = 14)**
- Powerplay runs and wickets for both teams
- Wickets in hand (both teams)
- Current run rate, required run rate
- Engineered: form difference, batting average difference, bowling economy difference, powerplay run rate, powerplay wicket pressure, run rate vs required, wicket momentum, powerplay advantage

## Training

- Optimiser: Adam with weight decay 1e-4
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Gradient clipping: max norm 1.0
- Early stopping: patience 12 (cross-validation), 15 (final model)
- Validation: 5-fold StratifiedKFold on training set before final model fit

## Usage

```bash
pip install -r requirements.txt
python main.py
```

This generates the dataset, trains with cross-validation, evaluates on the hold-out test set, and saves all plots and the model to `results/`.

## Top Predictive Features

From permutation importance (AUC drop on shuffling):

1. `pp_wickets_t1` (powerplay wickets taken by team 1)
2. `pp_wicket_pressure` (engineered: pp wickets normalised by overs)
3. `t2_wickets_in_hand`
4. `t1_bowling_economy`
5. `wickets_in_hand_t1`
6. `t1_form_win_pct`
7. `form_diff`
8. `pitch_encoded`

Powerplay wickets dominate because early wicket bursts in T20 are the strongest indicator of first innings control.

## Data

Synthetic data generated for 3000 T20 matches across 10 IPL teams and 10 venues. Match outcomes are derived from a domain-weighted logistic function combining all features with calibrated noise to simulate realistic uncertainty. Class balance: Team 1 wins ~70% (reflects home or batting-first advantage in the simulation).

## Requirements

- Python 3.9+
- PyTorch 2.0+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+
- matplotlib 3.7+
- seaborn 0.12+
