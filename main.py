from model import train_model
from predict import predict_today_games

if __name__ == "__main__":
    print("Training Model...")
    train_model()

    print("\nFetching NBA Games and Predicting...")
    predict_today_games()
