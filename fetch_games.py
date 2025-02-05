from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime

def get_today_games():
    today = datetime.now().strftime('%Y-%m-%d')
    games = scoreboard.ScoreBoard().get_dict()['scoreboard']['games']

    today_games = []
    for game in games:
        today_games.append({
            "home_team": game['homeTeam']['teamName'],
            "home_team_id": game['homeTeam']['teamId'],
            "away_team": game['awayTeam']['teamName'],
            "away_team_id": game['awayTeam']['teamId']
        })
    
    return today_games

if __name__ == "__main__":
    games = get_today_games()
    for g in games:
        print(f"{g['away_team']} at {g['home_team']}")
