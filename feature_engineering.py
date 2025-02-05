from data_processing import get_team_past_games

def compute_team_features(team_id, num_games=5):
    """Fetch past games and compute team stats."""
    df = get_team_past_games(team_id, num_games)

    if df.empty:
        print(f"⚠️ No game data found for team ID {team_id}")
        return None, None, None

    # Print available columns for debugging
    print(f"📊 Columns in dataset for team {team_id}: {df.columns.tolist()}")

    try:
        avg_points_scored = df['PTS'].mean()

        # Automatically find the correct opponent points column
        opponent_points_col = None
        for col in df.columns:
            if "OPP" in col.upper() and "PTS" in col.upper():
                opponent_points_col = col
                break

        if opponent_points_col:
            avg_points_allowed = df[opponent_points_col].mean()
        else:
            print(f"❌ Opponent points column not found for team ID {team_id}")
            return None, None, None

        # Estimate possessions using a standard formula
        if all(col in df.columns for col in ['FGA', 'TO', 'FTA', 'OREB']):
            df['Possessions'] = df['FGA'] + df['TO'] + 0.44 * df['FTA'] - df['OREB']
            avg_pace = 48 * df['Possessions'].mean() / 48
        else:
            avg_pace = None
            print(f"⚠️ Missing possession-related stats for team {team_id}")

        print(f"✅ Team {team_id} - Avg Points: {avg_points_scored}, Avg Opp Def: {avg_points_allowed}, Pace: {avg_pace}")
        return avg_points_scored, avg_points_allowed, avg_pace

    except Exception as e:
        print(f"❌ Error computing features for team ID {team_id}: {e}")
        return None, None, None
