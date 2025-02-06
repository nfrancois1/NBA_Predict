from data_processing import get_team_past_games
import pandas as pd

def compute_team_features(team_id, num_games=5):
    """Compute team features including avg points scored, opponent points allowed, and pace."""
    df = get_team_past_games(team_id, num_games)

    if df.empty:
        print(f"⚠️ No game data found for team ID {team_id}")
        return None, None, None

    try:
        avg_points_scored = df['PTS'].mean()

        # Extract opponent points using MATCHUP column
        df['Opponent_Team_ID'] = df['MATCHUP'].apply(lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else None)

        opponent_dfs = []
        for opp_id in df['Opponent_Team_ID'].dropna().unique():
            opp_df = get_team_past_games(int(opp_id), num_games)
            if not opp_df.empty:
                opp_df = opp_df[['Game_ID', 'PTS']].rename(columns={'PTS': 'OPP_PTS'})
                opponent_dfs.append(opp_df)

        if not opponent_dfs:
            print(f"❌ No opponent stats found for team ID {team_id}")
            return None, None, None

        opponent_df = pd.concat(opponent_dfs)
        df = df.merge(opponent_df, on='Game_ID', how='left')

        avg_points_allowed = df['OPP_PTS'].mean()

        # Compute estimated possessions
        if all(col in df.columns for col in ['FGA', 'TO', 'FTA', 'OREB']):
            df['Possessions'] = df['FGA'] + df['TO'] + 0.44 * df['FTA'] - df['OREB']
            avg_pace = df['Possessions'].mean()
        else:
            avg_pace = None
            print(f"⚠️ Missing possession-related stats for team {team_id}")

        return avg_points_scored, avg_points_allowed, avg_pace

    except Exception as e:
        print(f"❌ Error computing features for team ID {team_id}: {e}")
        return None, None, None
