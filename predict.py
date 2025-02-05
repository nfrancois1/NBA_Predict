import joblib
import numpy as np
from feature_engineering import compute_team_features
from fetch_games import get_today_games

def predict_today_games():
    model = joblib.load("nba_model.pkl")  # Load trained model
    games = get_today_games()

    for game in games:
        home_features = compute_team_features(game['home_team_id'])
        away_features = compute_team_features(game['away_team_id'])

        input_features = np.array(home_features + away_features).reshape(1, -1)
        predicted_points = model.predict(input_features)[0]

        print(f"{game['away_team']} at {game['home_team']} - Predicted Total Points: {predicted_points:.2f}")

if __name__ == "__main__":
    predict_today_games()
