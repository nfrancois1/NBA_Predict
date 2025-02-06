import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from feature_engineering import compute_team_features

NBA_TEAMS = {
    1610612737: "Atlanta Hawks",
    1610612738: "Boston Celtics",
    1610612739: "Cleveland Cavaliers",
    # Add all NBA team IDs here...
}

def create_dataset():
    """Generate training dataset."""
    data = []
    
    for team_id in NBA_TEAMS.keys():
        avg_scored, avg_allowed, pace = compute_team_features(team_id)
        if avg_scored is not None and avg_allowed is not None and pace is not None:
            data.append([team_id, avg_scored, avg_allowed, pace])

    if not data:
        print("❌ No valid data collected. Check `compute_team_features()`")
        return None

    df = pd.DataFrame(data, columns=['Team_ID', 'Avg_Scored', 'Avg_Allowed', 'Pace'])
    df.to_csv("nba_training_data.csv", index=False)
    print("✅ Dataset saved as `nba_training_data.csv`")
    return df

def train_model():
    """Train a model on NBA team stats to predict total points in a game."""
    data_path = "nba_training_data.csv"

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("❌ `nba_training_data.csv` not found. Generating dataset...")
        df = create_dataset()
        if df is None:
            return

    X = df[['Avg_Scored', 'Avg_Allowed', 'Pace']]
    y = df['Avg_Scored'] + df['Avg_Allowed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"📈 Model trained! MAE: {mae:.2f}")

    joblib.dump(model, "nba_model.pkl")
    print("✅ Model saved as `nba_model.pkl`")

def predict_game(team1_id, team2_id):
    """Predict the total points scored in a game between two teams."""
    model = joblib.load("nba_model.pkl")

    team1_features = compute_team_features(team1_id)
    team2_features = compute_team_features(team2_id)

    if None in team1_features or None in team2_features:
        print("❌ Missing data for one or both teams.")
        return None

    avg_scored = (team1_features[0] + team2_features[0]) / 2
    avg_allowed = (team1_features[1] + team2_features[1]) / 2
    pace = (team1_features[2] + team2_features[2]) / 2

    X_input = np.array([[avg_scored, avg_allowed, pace]])
    prediction = model.predict(X_input)

    print(f"🔮 Predicted total points: {prediction[0]:.2f}")
    return prediction[0]
