from nba_api.stats.endpoints import teamgamelog
import pandas as pd

def get_team_past_games(team_id, num_games=5):
    """Fetch past games for a given NBA team."""
    try:
        gamelog = teamgamelog.TeamGameLog(team_id=team_id)
        df = gamelog.get_data_frames()[0]

        if df.empty:
            print(f"⚠️ No game data found for team ID {team_id}")
            return pd.DataFrame()

        # Print available columns for debugging
        print(f"📊 Available columns in game log for team ID {team_id}: {df.columns.tolist()}")

        # Convert date to datetime format
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

        # Sort by date and select last 'num_games' games
        df = df.sort_values(by='GAME_DATE', ascending=False).head(num_games)

        return df

    except Exception as e:
        print(f"❌ Error fetching games for team ID {team_id}: {e}")
        return pd.DataFrame()
