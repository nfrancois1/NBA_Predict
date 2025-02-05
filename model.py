import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from nba_api.stats.static import teams
from feature_engineering import compute_team_features

def generate_training_data(output_file="nba_training_data.csv", num_games=10):
    """Fetch past games and generate a dataset."""
    from nba_api.stats.static import teams
    from feature_engineering import compute_team_features
    
    all_teams = teams.get_teams()
    data = []

    print("\n🔍 Fetching NBA team data...")
    for team in all_teams:
        team_id = team['id']
        print(f"📊 Processing {team['full_name']} (ID: {team_id})...")

        try:
            team_features = compute_team_features(team_id, num_games)
            
            if None in team_features or team_features == (None, None, None):  
                print(f"⚠️ Skipping {team['full_name']} - Missing stats")
                continue

            team_data = {
                "Team_Avg_Points": team_features[0],
                "Opponent_Avg_Defense": team_features[1],
                "Home_Away": 1,  # Placeholder
                "Total_Points": team_features[0] + team_features[1]  # Approximation
            }
            
            data.append(team_data)
            print(f"✅ Added: {team_data}")

        except Exception as e:
            print(f"❌ Error processing {team['full_name']}: {e}")
            continue  # Skip teams with errors

    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Training data saved to {output_file}")
    else:
        print("\n❌ No data collected. Check `compute_team_features()` for issues.")


def train_model(data_path="nba_training_data.csv"):
    """Train the NBA total points prediction model."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"{data_path} not found. Generating dataset...")
        generate_training_data()
        df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError("Generated training dataset is empty. Check `generate_training_data()` for issues.")

    X = df.drop(columns=['Total_Points'])
    y = df['Total_Points']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"✅ Model Trained. MAE: {mae}")

    joblib.dump(model, "nba_model.pkl")
    print("Model saved as nba_model.pkl")

if __name__ == "__main__":
    train_model()
