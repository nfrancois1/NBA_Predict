import requests
import pandas as pd
import time

NBA_STATS_API_URL = "https://stats.nba.com/stats/teamgamelogs"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9"
}

def fetch_team_game_logs(team_id, season="2023-24"):
    """Fetch past game logs for a given NBA team."""
    params = {
        "Season": season,
        "SeasonType": "Regular Season",
        "TeamID": team_id
    }

    try:
        response = requests.get(NBA_STATS_API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert response data into DataFrame
        columns = data["resultSets"][0]["headers"]
        rows = data["resultSets"][0]["rowSet"]
        df = pd.DataFrame(rows, columns=columns)

        # Ensure proper datetime format
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching games for team ID {team_id}: {e}")
        return pd.DataFrame()

def get_team_past_games(team_id, num_games=5):
    """Retrieve the last `num_games` games for the given team."""
    df = fetch_team_game_logs(team_id)
    if df.empty:
        return df
    return df.sort_values(by="GAME_DATE", ascending=False).head(num_games)
