"""
generate_data.py
Generates synthetic T20 cricket match dataset with realistic pre-match
and in-play features for match outcome prediction.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

TEAMS = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
]

VENUES = [
    'Wankhede Stadium', 'Chepauk', 'Chinnaswamy Stadium', 'Eden Gardens',
    'Arun Jaitley Stadium', 'Rajiv Gandhi Intl', 'Sawai Mansingh', 'PCA Stadium',
    'BRSABVE Ekana', 'Narendra Modi Stadium'
]

PITCH_TYPES = ['batting', 'balanced', 'bowling']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n=N):
    records = []

    team_strengths = {t: np.random.uniform(0.35, 0.65) for t in TEAMS}

    for i in range(n):
        # Teams
        team1, team2 = np.random.choice(TEAMS, size=2, replace=False)

        # Pre-match form (last 10 matches win %)
        t1_form = np.clip(np.random.normal(team_strengths[team1], 0.12), 0.1, 0.9)
        t2_form = np.clip(np.random.normal(team_strengths[team2], 0.12), 0.1, 0.9)

        # Head-to-head ratio (team1 wins / total h2h matches)
        h2h_ratio = np.clip(np.random.normal(0.5, 0.18), 0.1, 0.9)

        # Recent batting average (runs per match, last 5)
        t1_bat_avg = np.random.normal(162, 18)
        t2_bat_avg = np.random.normal(162, 18)

        # Recent bowling economy (runs conceded per over, last 5)
        t1_bowl_econ = np.random.normal(8.4, 0.9)
        t2_bowl_econ = np.random.normal(8.4, 0.9)

        # Venue stats
        venue = np.random.choice(VENUES)
        t1_venue_win_pct = np.clip(np.random.normal(0.5, 0.15), 0.05, 0.95)

        # Pitch type
        pitch = np.random.choice(PITCH_TYPES, p=[0.4, 0.4, 0.2])
        pitch_enc = {'batting': 1.0, 'balanced': 0.0, 'bowling': -1.0}[pitch]

        # Toss
        toss_won = np.random.randint(0, 2)
        toss_decision_bat = np.random.randint(0, 2)

        # In-play: team1 bats first (simplification)
        # Powerplay (overs 1-6)
        pp_runs = int(np.clip(np.random.normal(52, 10), 20, 90))
        pp_wickets = int(np.clip(np.random.poisson(1.4), 0, 5))

        # Mid-innings (overs 7-15)
        mid_runs = int(np.clip(np.random.normal(70, 12), 30, 110))
        mid_wickets = int(np.clip(np.random.poisson(2.5), 0, 6))

        # Wickets in hand at innings halfway
        wickets_in_hand = max(0, 10 - pp_wickets - mid_wickets)

        # Current run rate at over 15
        total_at_15 = pp_runs + mid_runs
        crr_at_15 = total_at_15 / 15.0

        # Final score team1
        overs_remaining = 5
        scoring_rate = np.random.normal(9.5, 1.5) * (1 + 0.05 * wickets_in_hand)
        final_score_t1 = int(total_at_15 + scoring_rate * overs_remaining)

        # Team2 chase features
        target = final_score_t1 + 1
        t2_powerplay_runs = int(np.clip(np.random.normal(50, 11), 18, 88))
        t2_powerplay_wkts = int(np.clip(np.random.poisson(1.5), 0, 5))
        t2_wickets_in_hand = max(0, 10 - t2_powerplay_wkts)

        # Required run rate after powerplay
        runs_remaining = target - t2_powerplay_runs
        overs_remaining_t2 = 14.0
        rrr = runs_remaining / overs_remaining_t2 if overs_remaining_t2 > 0 else 15.0

        # Compute outcome probability via logistic model with domain weights
        logit = (
            1.8 * (t1_form - t2_form)
            + 0.9 * h2h_ratio
            + 0.012 * (t1_bat_avg - t2_bat_avg)
            - 0.15 * (t1_bowl_econ - t2_bowl_econ)
            + 0.6 * t1_venue_win_pct
            + 0.3 * toss_won
            + 0.4 * pitch_enc
            + 0.04 * pp_runs
            - 0.5 * pp_wickets
            + 0.02 * crr_at_15
            + 0.3 * wickets_in_hand
            - 0.25 * rrr
            + 0.03 * (t2_powerplay_runs - 50)
            - 0.4 * t2_powerplay_wkts
            + np.random.normal(0, 0.5)
        )
        prob_t1_wins = sigmoid(logit)
        outcome = int(np.random.binomial(1, prob_t1_wins))

        records.append({
            'match_id': f'T20-{i+1:04d}',
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'pitch_type': pitch,
            't1_form_win_pct': round(t1_form, 3),
            't2_form_win_pct': round(t2_form, 3),
            'h2h_ratio': round(h2h_ratio, 3),
            't1_batting_avg': round(t1_bat_avg, 1),
            't2_batting_avg': round(t2_bat_avg, 1),
            't1_bowling_economy': round(t1_bowl_econ, 2),
            't2_bowling_economy': round(t2_bowl_econ, 2),
            't1_venue_win_pct': round(t1_venue_win_pct, 3),
            'toss_won_by_t1': toss_won,
            'toss_decision_bat': toss_decision_bat,
            'pp_runs_t1': pp_runs,
            'pp_wickets_t1': pp_wickets,
            'mid_runs_t1': mid_runs,
            'mid_wickets_t1': mid_wickets,
            'wickets_in_hand_t1': wickets_in_hand,
            'crr_at_over15': round(crr_at_15, 2),
            'final_score_t1': final_score_t1,
            't2_pp_runs': t2_powerplay_runs,
            't2_pp_wickets': t2_powerplay_wkts,
            't2_wickets_in_hand': t2_wickets_in_hand,
            'required_run_rate': round(rrr, 2),
            'outcome': outcome,
        })

    df = pd.DataFrame(records)
    df.to_csv('data/matches.csv', index=False)
    print(f"Generated {len(df)} matches | Win rate team1: {df['outcome'].mean():.3f}")
    print(f"Saved to data/matches.csv")
    return df


if __name__ == '__main__':
    generate_dataset()
