# Aviation_incident_classification

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

All four models use TF-IDF vectorization on the Narrative Text column, with a stratified 70/15/15 train/dev/test split (`random_state=42` throughout).

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.79 | 0.62 | 0.80 |
| Naive Bayes | 0.79 | 0.59 | 0.79 |
| SVM | 0.82 | 0.65 | 0.82 |
| Neural Network (w/o inverse-frequency weighting) | 0.82 | 0.64 | 0.82 |
| Neural Network (w/ inverse-frequency weighting) | 0.80 | 0.64 | 0.81 |

### Key Findings
- Equipment Problem consistently achieved the highest F1 (0.91–0.92) across all models, reflecting its dominance in the dataset (~55% of records)
- Ground Event was the most challenging class for every model, driven by both limited training examples (229 records) and genuine label ambiguity — some incidents are tagged as both Conflict and Ground Event by ASRS staff themselves
- SVM shows a notable overfitting gap (train accuracy 99.1% vs. dev accuracy 82.1%), growing steadily as its regularization parameter (`C`) increases
- Inverse-frequency weighting in the neural network nearly doubles Ground Event's F1 (0.16 → 0.29) but doesn't improve overall macro F1 — the gain is offset by small F1 drops across most other classes, a more nuanced trade-off than a simple "better balance for free"

## API (Work in Progress)

This section will cover the deployed inference API being built around the trained models — endpoints, request/response format, local setup, and the live deployment URL once available.

**Planned:**
- `POST /predict` — submit an incident narrative, receive a predicted category
- Containerized with Docker
- Deployed to a live public URL

*Check back soon, or see the commit history for current progress.*

## Requirements
```
pandas
numpy
scikit-learn
tensorflow / keras
```

## Acknowledgments
Original team project: Mohammed Emaad Hazari, Tom Cu, Griffen Lee, Mohammed Zubair — Loyola University Chicago, NLP Course (COMP 329). This repository extends that work with an individually-built and deployed API layer.

Dataset Source: https://asrs.arc.nasa.gov/search/database.html
