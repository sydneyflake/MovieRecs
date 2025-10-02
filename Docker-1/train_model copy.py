"""
Model Training Script
Train the LensKit BiasedMF model and save it for use in the app.
"""

import pandas as pd
import pickle
import pathlib
from lenskit.algorithms.als import BiasedMF
from lenskit import Recommender
from pathlib import Path
import json
import numpy as np
from lenskit import batch
from lenskit.metrics.predict import rmse
import json, os, tempfile
from sklearn.metrics import mean_squared_error, mean_absolute_error



def load_data():
    """Load and prepare the MovieLens data."""
    print("Loading data...")
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv')
    return ratings, movies

def normalize_title(title: str) -> str:
    parts = title.rsplit("(", 1)  # separate year if present
    name = parts[0].strip()
    year = "(" + parts[1] if len(parts) > 1 else ""

    for article in [" The", " A", " An"]:
        suffix = f", {article}"
        if name.endswith(suffix):
            name = f"{article} {name[: -len(suffix)]}"

    return f"{name.strip()} {year}".strip()

def prepare_training_data(ratings):
    """Prepare data in LensKit format."""
    print("Preparing training data...")
    train = ratings.rename(columns={
        'userId': 'user', 
        'movieId': 'item'
    })[['user', 'item', 'rating']].copy()
    return train

def train_model(train_data, features=50, iterations=40, reg=0.02, damping=5.0):
    """Train the BiasedMF model."""
    print("Training model...")
    mf = BiasedMF(features, iterations=iterations, reg=reg, damping=damping)
    rec = Recommender.adapt(mf).fit(train_data)
    print("Model training complete!")
    return rec

def save_model(model, path='models/lenskit_model.pkl'):
    """Save the trained model to disk."""
    print(f"Saving model to {path}...")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")


def evaluate_model(train_frame):
    """
    Simple 80/20 hold-out evaluation.
    Returns: rmse_val, mse_val
    """
    # 80/20 split
    test = train_frame.sample(frac=0.2, random_state=42)
    train = train_frame.drop(test.index)

    # train fresh model on train split
    mf = BiasedMF(50, iterations=40, reg=0.02, damping=5.0)
    rec = Recommender.adapt(mf).fit(train)

    # predict test ratings
    preds = batch.predict(rec, test)  # columns: user, item, prediction
    merged = preds.merge(test, on=["user", "item"], how="inner")  # adds 'rating'

    # metrics
    rmse_val = float(mean_absolute_error(merged["prediction"], merged["rating"], squared = False))
    mse_val  = float(mean_squared_error((merged["prediction"] - merged["rating"])))
    return rmse_val, mse_val


def main():
    ratings, movies = load_data()
    train = prepare_training_data(ratings)
    rmse_val, mse_val = evaluate_model(train)

    # Train final model on full data
    model = train_model(train)
    save_model(model)

    Path("models").mkdir(parents=True, exist_ok=True)
    movies.to_csv("models/movies.csv", index=False)

    # ---- Save metrics to a text file instead of JSON ----
    metrics_path = Path("models/metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse_val:.4f}\n")
        f.write(f"MSE:  {mse_val:.4f}\n")

    print(f"\nMetrics saved to {metrics_path}")
    print("Training complete! Model ready for use.")



if __name__ == "__main__":
    main()