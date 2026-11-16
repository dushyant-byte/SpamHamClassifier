"""
Spam Classifier Full Project
Models: Linear Regression, Logistic Regression, Decision Tree, Random Forest, SVM, KNN
Features: Confusion Matrix, Precision, Recall, Accuracy
Data saved as JSON + Flask Dashboard
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

try:  # charts are optional - training must still work without them
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS = True
except Exception as _e:
    print(f"[!] matplotlib/seaborn not available ({_e}); skipping charts.")
    PLOTS = False

# ==================== CONFIG ====================
# Always run relative to this file's folder, whatever the current directory is
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "SMSSpamCollection"
OUTPUT_DIR = "results"
MODELS_DIR = "models"
JSON_PATH = "results/model_results.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# ==================== LOAD DATA ====================
print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

# Encode labels: ham=0, spam=1
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# ==================== PREPROCESSING ====================
print("\nVectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(df['message'])
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Save vectorizer
with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

# ==================== MODELS ====================
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=20),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

results = {}
all_predictions = {}

print("\nTraining models...")
for name, model in models.items():
    print(f"  -> Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    if name == "Linear Regression":
        # Continuous predictions -> threshold at 0.5
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
    
    all_predictions[name] = y_pred.tolist()
    
    # Save model
    model_filename = name.lower().replace(" ", "_") + ".pkl"
    with open(os.path.join(MODELS_DIR, model_filename), "wb") as f:
        pickle.dump(model, f)
    
    # Plot Confusion Matrix
    if PLOTS:
        try:
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            img_path = f"static/images/cm_{name.lower().replace(' ', '_')}.png"
            plt.savefig(img_path, dpi=100)
            plt.close()
        except Exception as _e:
            print(f"     [!] chart skipped: {_e}")
    print(f"     Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

# ==================== COMPARISON PLOT ====================
if PLOTS:
    try:
        print("\nGenerating comparison charts...")
        metrics_df = pd.DataFrame({
            name: {
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1-Score": res["f1_score"]
            } for name, res in results.items()
        }).T

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title("Model Performance Comparison", fontsize=14, fontweight='bold')
        plt.ylabel("Score")
        plt.xlabel("Models")
        plt.xticks(rotation=30, ha='right')
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/images/model_comparison.png", dpi=120)
        plt.close()

        # Accuracy only bar
        plt.figure(figsize=(10, 5))
        accs = [results[m]["accuracy"] for m in models.keys()]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
        bars = plt.bar(list(models.keys()), accs, color=colors)
        plt.title("Accuracy Comparison of All Models", fontsize=14, fontweight='bold')
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30, ha='right')
        plt.ylim(0.8, 1.0)
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig("static/images/accuracy_comparison.png", dpi=120)
        plt.close()
    except Exception as _e:
        print(f"[!] comparison charts skipped: {_e}")

# ==================== SAVE JSON ====================
output_data = {
    "project": "Spam SMS Classifier",
    "dataset": "UCI SMS Spam Collection",
    "total_samples": int(len(df)),
    "ham_count": int((df['label'] == 'ham').sum()),
    "spam_count": int((df['label'] == 'spam').sum()),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "features": 3000,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "models": results,
    "best_model": max(results, key=lambda x: results[x]["accuracy"]),
    "label_mapping": {"0": "ham", "1": "spam"}
}

with open(JSON_PATH, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"\nResults saved to {JSON_PATH}")
print(f"Best Model: {output_data['best_model']} "
      f"(Accuracy: {results[output_data['best_model']]['accuracy']})")

# Also save a summary CSV
summary = []
for name, res in results.items():
    summary.append({
        "Model": name,
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1-Score": res["f1_score"],
        "TN": res["tn"],
        "FP": res["fp"],
        "FN": res["fn"],
        "TP": res["tp"]
    })
pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)

print("\n=== TRAINING COMPLETE ===")
print("All models trained, confusion matrices saved, JSON ready.")
print("Run 'python app.py' to start the Flask dashboard.")
