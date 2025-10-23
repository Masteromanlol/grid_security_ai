#!/usr/bin/env python
# train_model.py
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

def train_model(data_path, model_output_path):
    """
    Trains a machine learning model on the aggregated simulation data.

    Args:
        data_path (str): The path to the aggregated data file.
        model_output_path (str): The path to save the trained model.
    """
    # Load the data
    df = pd.read_pickle(data_path)

    # Preprocess the data
    # For simplicity, we'll just use the success status as the target
    # and some basic features from the contingency
    df['target'] = df['success'].astype(int)
    df['contingency_type'] = df['contingency'].apply(lambda x: x['type'])
    df['contingency_id'] = df['contingency'].apply(lambda x: x['id'])

    features = pd.get_dummies(df[['contingency_type', 'contingency_id']], columns=['contingency_type'])
    target = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on simulation data.")
    parser.add_argument("data_path", type=str, help="Path to the aggregated data file.")
    parser.add_argument("model_output_path", type=str, help="Path to save the trained model.")
    args = parser.parse_args()

    train_model(args.data_path, args.model_output_path)
