# SMS Spam Classifier - Full ML Project

## Overview
Complete spam classification project using **6 machine learning models**:

1. **Linear Regression** (with 0.5 threshold)
2. **Logistic Regression**
3. **Decision Tree**
4. **Random Forest**
5. **Support Vector Machine (SVM)**
6. **K-Nearest Neighbors (KNN)**

## Features
- TF-IDF vectorization (3000 features, unigrams + bigrams)
- Confusion Matrix for every model
- Accuracy, Precision, Recall, F1-Score
- Results saved in JSON (`results/model_results.json`)
- Beautiful Flask Dashboard with live prediction
- Comparison charts

## Dataset
- **UCI SMS Spam Collection**
- 5572 messages (4825 Ham + 747 Spam)

## Results (Test Set)

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Linear Regression  | 0.9587   | 0.8411    | 0.8523 | ~0.85    |
| Logistic Regression| 0.9713   | 1.0000    | 0.7852 | ~0.88    |
| Decision Tree      | 0.9489   | 0.8433    | 0.7584 | ~0.80    |
| Random Forest      | 0.9776   | 1.0000    | 0.8322 | ~0.91    |
| **SVM (Best)**     | **0.9821** | 0.9924  | 0.8725 | **~0.93**|
| KNN                | 0.9148   | 1.0000    | 0.3624 | ~0.53    |

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (already done if you have results/)
python spam_classifier.py

# 3. Start Flask Dashboard
python app.py
```

Open browser: **http://127.0.0.1:5000**

## Project Structure
```
spam_classifier/
├── SMSSpamCollection          # Dataset
├── spam_classifier.py         # Training script
├── app.py                     # Flask dashboard
├── requirements.txt
├── models/                    # Saved .pkl models + vectorizer
├── results/
│   ├── model_results.json     # All metrics in JSON
│   └── metrics_summary.csv
├── static/images/             # Confusion matrices + charts
└── templates/
    ├── index.html             # Main dashboard
    └── compare.html
```

## Live Prediction
Dashboard allows you to:
- Select any of the 6 models
- Type any SMS message
- Get instant Spam / Ham prediction with confidence score
