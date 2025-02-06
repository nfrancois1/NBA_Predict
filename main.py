from model import train_model, predict_game

if __name__ == "__main__":
    train_model()
    
    team1 = 1610612737  # Atlanta Hawks
    team2 = 1610612738  # Boston Celtics

    predict_game(team1, team2)
