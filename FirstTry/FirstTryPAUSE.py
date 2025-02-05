import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ Load dataset
df = pd.read_csv(r'NBA_Predict/csv/TeamStatistics.csv')

# ✅ Rename columns
df = df.rename(columns={
    'teamName': 'Team',
    'opponentTeamName': 'Opponent',
    'teamScore': 'Points',
    'opponentScore': 'Opponent_Points',
    'home': 'Home_Away'
})

# ✅ Compute rolling averages
df['Team_Avg_Points'] = df.groupby('Team')['Points'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['Opponent_Avg_Defense'] = df.groupby('Opponent')['Opponent_Points'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# ✅ Fix Home/Away Mapping
df['Home_Away'] = df['Home_Away'].map({True: 1, False: 0})

# ✅ Fix FutureWarning: Replace fillna(inplace=True)
df = df.assign(
    Team_Avg_Points=df['Team_Avg_Points'].fillna(df['Team_Avg_Points'].median()),
    Opponent_Avg_Defense=df['Opponent_Avg_Defense'].fillna(df['Opponent_Avg_Defense'].median()),
    Home_Away=df['Home_Away'].fillna(0)
)

# ✅ Define features and target
features = ['Team_Avg_Points', 'Opponent_Avg_Defense', 'Home_Away']
target = 'Points'

X = df[features]
y = df[target]

# ✅ Debug: Print final dataset shape
print(f"Final dataset shape: {X.shape}, Target shape: {y.shape}")

# ✅ Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test)

# ✅ Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')

# ✅ Example Prediction for a New Game
new_game = pd.DataFrame({
    'Team_Avg_Points': [110],
    'Opponent_Avg_Defense': [105],
    'Home_Away': [1]
})

# ✅ Ensure correct feature order before scaling
new_game = new_game[features]
new_game = scaler.transform(new_game)
predicted_points = model.predict(new_game)
print(f'Predicted Total Points: {predicted_points[0]:.2f}')
