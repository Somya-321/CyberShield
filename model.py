import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re


# --- URL Model Training ---
def load_url_dataset(file_path):
    """Load the URL dataset from CSV."""
    df = pd.read_csv(file_path)
    # Drop 'index' column if present
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    # Map Result: -1 (phishing) -> 0, 1 (legitimate) -> 1
    df['Result'] = df['Result'].map({-1: 0, 1: 1})
    X = df.drop(columns=['Result'])
    y = df['Result']
    return X, y


def train_url_model(file_path):
    """Train the URL phishing model using the provided dataset."""
    # Load dataset
    X, y = load_url_dataset(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define model and hyperparameter grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"URL Phishing Model Accuracy: {accuracy:.2f}")
    print("URL Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Phishing', 'Legitimate']))

    # Save model
    joblib.dump(best_model, 'phishing_model.pkl')
    print(f"URL Model Best Parameters: {grid_search.best_params_}")


# --- Email Model Training ---
def train_email_model(file_path):
    """Train the email phishing model using the provided dataset."""
    # Load email dataset
    email_data = pd.read_csv(file_path)

    # Handle missing values in 'Email Text'
    print("Checking for NaN values in 'Email Text'...")
    nan_count = email_data['Email Text'].isna().sum()
    print(f"Found {nan_count} NaN values in 'Email Text'.")

    # Replace NaN with empty string and preprocess
    email_data['Email Text'] = email_data['Email Text'].fillna("")
    email_data['Email Text'] = email_data['Email Text'].astype(str)
    email_data['Email Text'] = email_data['Email Text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # Features and target
    X = email_data['Email Text']
    y = email_data['Email Type']

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

    # Define model and hyperparameter grid
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Email Phishing Model Accuracy: {accuracy:.2f}")
    print("Email Model Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Phishing Email', 'Safe Email']))

    # Save model and vectorizer
    joblib.dump(best_model, 'email_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print(f"Email Model Best Parameters: {grid_search.best_params_}")


if __name__ == '__main__':
    print("Training URL Model...")
    train_url_model('dataset.csv')
    print("\nTraining Email Model...")
    train_email_model('Phishing_Email.csv')