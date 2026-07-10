"""
features.py
Feature engineering for cricket match outcome prediction.
Encodes categorical variables and scales numerical features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = [
    't1_form_win_pct', 't2_form_win_pct', 'h2h_ratio',
    't1_batting_avg', 't2_batting_avg',
    't1_bowling_economy', 't2_bowling_economy',
    't1_venue_win_pct',
    'toss_won_by_t1', 'toss_decision_bat',
    'pp_runs_t1', 'pp_wickets_t1',
    'mid_runs_t1', 'mid_wickets_t1',
    'wickets_in_hand_t1', 'crr_at_over15',
    't2_pp_runs', 't2_pp_wickets',
    't2_wickets_in_hand', 'required_run_rate',
]

ENGINEERED_FEATURES = [
    'form_diff', 'batting_avg_diff', 'bowling_econ_diff',
    'pp_run_rate_t1', 'pp_wicket_pressure',
    'run_rate_vs_required', 'wicket_momentum',
    'powerplay_advantage',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Differential features — domain insight: relative strength matters more than absolute
    df['form_diff'] = df['t1_form_win_pct'] - df['t2_form_win_pct']
    df['batting_avg_diff'] = df['t1_batting_avg'] - df['t2_batting_avg']
    df['bowling_econ_diff'] = df['t2_bowling_economy'] - df['t1_bowling_economy']  # lower is better

    # Powerplay run rate (runs per over, overs 1-6)
    df['pp_run_rate_t1'] = df['pp_runs_t1'] / 6.0

    # Wicket pressure index: wickets lost weighted by phase importance
    df['pp_wicket_pressure'] = df['pp_wickets_t1'] * 1.5 + df['mid_wickets_t1']

    # Run rate vs required at powerplay stage for chasing team
    df['run_rate_vs_required'] = df['crr_at_over15'] - df['required_run_rate']

    # Wicket momentum: wickets in hand difference between both teams at comparable stage
    df['wicket_momentum'] = df['wickets_in_hand_t1'] - df['t2_wickets_in_hand']

    # Powerplay comparison: team1 pp runs vs team2 pp runs
    df['powerplay_advantage'] = df['pp_runs_t1'] - df['t2_pp_runs']

    # Encode pitch type
    pitch_map = {'batting': 1, 'balanced': 0, 'bowling': -1}
    df['pitch_encoded'] = df['pitch_type'].map(pitch_map)

    return df


def get_feature_matrix(df: pd.DataFrame):
    df = engineer_features(df)
    all_features = NUMERICAL_FEATURES + ENGINEERED_FEATURES + ['pitch_encoded']
    X = df[all_features].values.astype(np.float32)
    y = df['outcome'].values.astype(np.int64)
    return X, y, all_features


def get_scaler(X_train: np.ndarray):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
