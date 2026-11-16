"""
Flask Dashboard for Spam Classifier  (fixed version)

Fixes:
  1. Absolute paths  -> works no matter which folder you run it from.
  2. Pickle/sklearn version mismatch -> models are auto-retrained with YOUR
     installed scikit-learn if the saved .pkl files can't be loaded safely.
  3. /predict never crashes with a bare "Internal Server Error": it returns
     a JSON error and prints the real traceback in the terminal.
"""

import json
import os
import pickle
import subprocess
import sys
import traceback
import warnings

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # spam_classifier.py & templates use relative paths

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "results", "model_results.json")

MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "KNN": "knn.pkl",
}

app = Flask(__name__)


def train_models():
    """Re-run the training script with the current Python / scikit-learn."""
    print("\n[!] Saved models are missing or were built with a different "
          "scikit-learn version.")
    print("[!] Re-training now (takes about 1 minute, only happens once)...\n")
    subprocess.check_call(
        [sys.executable, os.path.join(BASE_DIR, "spam_classifier.py")],
        cwd=BASE_DIR,
    )


def load_artifacts(strict=True):
    """Load vectorizer + models. Raises if anything is missing/incompatible."""
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    with warnings.catch_warnings():
        # Turn "pickle made with another sklearn version" into a real error
        warnings.simplefilter("error" if strict else "ignore")
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)

        models = {}
        for name, filename in MODEL_FILES.items():
            with open(os.path.join(MODELS_DIR, filename), "rb") as f:
                models[name] = pickle.load(f)

    # Smoke test so a broken model is caught at startup, not on first click
    X = tfidf.transform(["test message"])
    for m in models.values():
        m.predict(X)

    return results, tfidf, models


try:
    RESULTS, TFIDF, MODELS = load_artifacts()
except Exception as e:  # missing file, version mismatch, corrupt pickle...
    print(f"[!] Could not load saved models: {type(e).__name__}: {e}")
    try:
        train_models()
        RESULTS, TFIDF, MODELS = load_artifacts()
    except Exception as e2:
        print(f"\n[!] Auto-retrain failed ({type(e2).__name__}). Read the error printed above it.")
        print("[!] Trying to use the saved models anyway (ignoring version warning)...")
        RESULTS, TFIDF, MODELS = load_artifacts(strict=False)

print(f"Loaded {len(MODELS)} models: {', '.join(MODELS)}")


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html", results=RESULTS)


@app.route("/compare")
def compare():
    return render_template("compare.html", results=RESULTS)


@app.route("/api/results")
def api_results():
    return jsonify(RESULTS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    model_name = data.get("model", "Logistic Regression")

    if not message:
        return jsonify({"error": "Please enter a message"}), 400
    if model_name not in MODELS:
        return jsonify({"error": f"Model {model_name} not found"}), 400

    try:
        X = TFIDF.transform([message])
        model = MODELS[model_name]

        if model_name == "Linear Regression":
            pred_raw = float(np.ravel(model.predict(X))[0])
            pred = 1 if pred_raw >= 0.5 else 0
            proba = float(np.clip(pred_raw, 0, 1))
        else:
            pred = int(model.predict(X)[0])
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X)[0][1])
            else:
                proba = float(pred)

        label = "Spam" if pred == 1 else "Ham (Not Spam)"
        return jsonify({
            "message": message,
            "model": model_name,
            "prediction": label,
            "is_spam": bool(pred),
            "confidence": round(proba * 100, 2) if pred == 1 else round((1 - proba) * 100, 2),
            "spam_probability": round(proba * 100, 2),
        })
    except Exception as e:
        traceback.print_exc()  # real error shows in your terminal
        return jsonify({"error": f"Prediction failed: {type(e).__name__}: {e}"}), 500


@app.errorhandler(500)
def server_error(e):
    traceback.print_exc()
    return jsonify({"error": "Internal server error - check the terminal for details"}), 500


if __name__ == "__main__":
    print("Starting Spam Classifier Dashboard...")
    print("Open http://127.0.0.1:5000 in your browser")
    # use_reloader=False: otherwise Flask debug mode loads (and may retrain) twice
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
