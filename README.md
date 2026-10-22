# Aviation Incident Classification API

> **Note:** This project began as a 4-person team submission for a Natural Language Processing course (COMP 329, Loyola University Chicago). This repository contains my individual extension of that work — building and deploying a live API around the trained models. The original data sourcing, ETL, and Logistic Regression baseline were also my work; the SVM/Naive Bayes and neural network models were built by teammates as part of the original team project. 
## Overview
This project applies Natural Language Processing (NLP) techniques to automatically classify aviation incident reports from NASA's Aviation Safety Reporting System (ASRS). Given a free-text narrative written by a pilot or air traffic controller, the goal is to predict the type of incident that occurred.

## Dataset
- **Source:** [NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html)
- **Size:** ~15,900 records (filtered to commercial aircraft: Airbus and Boeing models)
- **Date Range:** December 2015 – December 2025
- **Input:** Free-text narrative reports written by pilots and controllers
- **Label:** Event type (6 categories)

## Event Labels
| Label | Description | Count |
|---|---|---|
| Equipment Problem | Aircraft system or mechanical failures (critical and less severe) | 8,671 |
| Deviation | Procedural violations or parameter exceedances (altitude, speed, heading) | 3,925 |
| ATC Issue | Air traffic control communication or clearance issues | 1,836 |
| Conflict | Ground conflicts, airborne conflicts, and near mid-air collisions (NMAC) | 822 |
| Inflight Event | In-flight encounters including wake vortex, weather, and loss of control | 410 |
| Ground Event | Ground operation incidents involving personnel, vehicles, or equipment | 229 |

## Data Preprocessing
- Combined 4 raw ASRS CSV exports (Dec 2015 – Dec 2025) using MS Excel Power Query, filtered to commercial aircraft models
- Consolidated 3,400+ raw event label combinations into 6 clean categories
- Dropped noise labels (e.g. "No Specific Anomaly Occurred") and labels with fewer than 3 examples
- Concatenated Report 1 + Report 2 into a single Narrative Text column, handling placeholder text and null values
- Applied lowercasing and punctuation removal

## Model Development

All models use TF-IDF vectorization on the Narrative Text column, with a stratified 70/15/15 train/dev/test split (`random_state=42` throughout). Results below are on the held-out test set.

| Model | Accuracy | Macro F1 | Weighted F1 | Ground Event F1 | Ground Event Recall |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.79 | 0.63 | 0.80 | 0.32 | 0.54 |
| Naive Bayes | 0.80 | 0.61 | 0.80 | 0.29 | 0.17 |
| SVM | 0.83 | 0.65 | 0.82 | 0.28 | 0.17 |
| Neural Network (w/o inverse-frequency weighting) | 0.82 | 0.64 | 0.82 | 0.20 | 0.11 |
| Neural Network (w/ inverse-frequency weighting) | 0.81 | 0.65 | 0.81 | 0.31 | 0.37 |

All runs are tracked in MLflow (parameters and metrics per model). Differences of ~0.01 are within run-to-run noise, and Ground Event has only 35 test samples, so its metrics are especially noisy.

### Key Findings
- Equipment Problem consistently achieved the highest F1 (0.91–0.92) across all models, reflecting its dominance in the dataset (~55% of records)
- Ground Event was the most challenging class for every model, driven by both limited training examples (229 records) and genuine label ambiguity — some incidents are tagged as both Conflict and Ground Event by ASRS staff themselves
- SVM shows a notable overfitting gap (train accuracy 99.1% vs. dev accuracy 82.1%), growing steadily as its regularization parameter (`C`) increases
- Class weighting helps the minority class, but model complexity doesn't. Inverse-frequency weighting raised the neural network's Ground Event recall from 0.11 to 0.37 (4 → 13 of 35 test samples) with little change in overall scores (macro F1 0.64 → 0.65, accuracy 0.82 → 0.81). The simpler Logistic Regression with balanced class weights does as well or better on Ground Event (F1 0.32, recall 0.54), so the neural network adds no clear advantage on this dataset. With only 35 test samples and single runs, these Ground Event differences should be read as directional, not conclusive

## API

A live REST API serves the trained models.

**Live demo:** https://aviation-incident-api.onrender.com/docs
> Hosted on Render's free tier, so the service sleeps when idle — the **first request can take 50+ seconds**. After that it responds quickly.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Classify a narrative with the Logistic Regression pipeline (TF-IDF + LR) |
| `/predict/nn` | POST | Classify a narrative with the neural network (served via ONNX Runtime) |
| `/report/{report_id}` | GET | Look up a stored report and its label from SQLite (404 if missing) |
| `/` | GET | Redirects to the interactive docs at `/docs` |

All prediction and report routes require an `X-API-Key` header (missing/wrong key → `401`, malformed input → `422`). The API is key-protected; to try it, email me for a demo key: **mohd.emaad10@gmail.com**.

```bash
curl -X POST https://aviation-incident-api.onrender.com/predict \
  -H "X-API-Key: <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Crew reported cabin pressure warning during climb."}'
```

### Engineering highlights
- **FastAPI + Pydantic** request validation; API-key auth via dependency injection, key loaded from environment variables (the app refuses to start without it)
- **Models loaded once at startup.** The LR model is a single scikit-learn `Pipeline` (vectorizer + classifier) saved with joblib, so training and serving preprocessing cannot drift apart
- **ONNX Runtime for the neural network:** the Keras model was exported to ONNX and wrapped in `NNclassifier`, which bundles the vectorizer, label encoder and ONNX session behind one `predict_text()` call. Replacing TensorFlow cut the Docker image from **903 MB to ~235 MB (~74% smaller)**; ONNX predictions were checked against the original Keras model on 1000 held-out samples and matched
- **Tests:** pytest suite (~13 tests, including parametrized cases) covering auth failures, validation errors, successful predictions, 404/200 report lookups and the root redirect
- **CI:** GitHub Actions on every push to `main` — install dependencies, build the database, run tests (API key via GitHub Secrets), build the Docker image
- **Docker:** `python:3.12-slim`, dependency layer cached separately from code, database built at image build time, `.dockerignore` keeps secrets and training artifacts out of the image
- **Deployment:** Docker image deployed on Render

### Run locally
```bash
git clone <this-repo>
cd <repo>
python -m venv venv && source venv/bin/activate   # Python 3.12
pip install -r requirements.txt
python build_connection.py                         # builds Aviation.db from clean_data.csv
echo "API_KEY=your-secret-key" > .env
uvicorn main:app --reload
```
Open http://localhost:8000/docs, click **Authorize**, and enter your key.

With Docker:
```bash
docker build -t aviation-api-onnx .
docker run -e API_KEY=your-secret-key -p 8000:8000 aviation-api-onnx
```

Run the tests: `API_KEY=your-secret-key pytest`

The API serves the Logistic Regression pipeline and the weighted neural network.

### Limitations
- Free-tier hosting means cold starts and limited memory (512 MB)
- Single shared API key; a real deployment would use per-user keys or JWT
- No health-check endpoint or production monitoring yet
- SQLite database is rebuilt from the CSV at image build time — fine for a demo, not for live data

## Requirements
```
pandas
numpy
scikit-learn
onnxruntime
fastapi
uvicorn
joblib
python-dotenv
pytest
httpx
```
(Training notebooks additionally use tensorflow/keras and mlflow. See `requirements.txt` for the pinned serving dependencies.)

## Acknowledgments
Original team project: Mohammed Emaad Hazari, Tom Cu, Griffen Lee, Mohammed Zubair — Loyola University Chicago, NLP Course (COMP 329). This repository extends that work with an individually-built and deployed API layer.

Dataset Source: https://asrs.arc.nasa.gov/search/database.html
