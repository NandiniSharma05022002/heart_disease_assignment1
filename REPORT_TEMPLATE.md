# MLOps Assignment 01 — Heart Disease Prediction

> Replace prompts with your own observed results and screenshots. Do not fabricate metrics, URLs or deployment evidence.

## 1. Executive Summary
- Problem:
- Dataset:
- Best model:
- Test ROC-AUC:
- Deployment approach:
- Monitoring approach:

## 2. Problem Statement and Dataset
Explain the binary classification objective and UCI Heart Disease dataset. Describe the conversion of original disease severity values into binary presence/absence.

## 3. Data Acquisition and Cleaning
- Acquisition method:
- Raw file:
- Missing-value handling:
- Target transformation:
- Final row/feature counts:

## 4. EDA
Insert the generated EDA figures and discuss at least three observations:
- `reports/figures/feature_distributions.png`
- `reports/figures/class_balance.png`
- `reports/figures/correlation_heatmap.png`

## 5. Feature Engineering
Explain the sklearn pipeline:
- numerical features: median imputation + standardization;
- categorical features: most-frequent imputation + one-hot encoding;
- transformations are fitted inside the pipeline to avoid preprocessing leakage.

## 6. Model Development
Compare:
- Logistic Regression (`C=1.0`);
- Random Forest (`n_estimators=300`, `min_samples_leaf=2`, balanced class weights).

Include:
- the final hyperparameter configurations and why they were retained;
- 5-fold stratified CV results;
- holdout accuracy;
- precision;
- recall;
- ROC-AUC;
- model-selection rationale.

## 7. Experiment Tracking
Insert an MLflow screenshot showing both runs, parameters, metrics and artifacts.

Explain:
- experiment name;
- SQLite tracking backend;
- run IDs;
- logged plots/model artifacts;
- reproducibility benefits.

## 8. Reusable Model Packaging
Explain that the selected **complete fitted sklearn Pipeline** is saved as:

```text
artifacts/model.pkl
```

Also discuss:

```text
artifacts/model_metadata.json
artifacts/model_comparison.csv
```

Mention that pickle is trusted-only serialization and that MLflow remains responsible for experiment/model tracking.

## 9. API
Document:

- `GET /health`
- `POST /predict`
- `GET /metrics`
- `GET /docs`

Insert successful API screenshots.

## 10. Containerization
Explain the Docker image, exposed port 8000, model artifact inclusion and Docker Compose stack.

Include evidence of:

- API container;
- Prometheus container;
- Grafana container;
- successful prediction.

## 11. Kubernetes Deployment
Explain:

- Minikube cluster;
- Deployment with 2 replicas;
- readiness/liveness probes;
- resource requests/limits;
- `LoadBalancer` Service.

Insert:

- `kubectl get pods -n heart-disease`;
- `kubectl get service -n heart-disease`;
- successful `/health` and `/predict` calls through the service URL.

## 12. Monitoring and Observability
Explain the following metrics:

- `api_requests_total` — request count by endpoint/method/status;
- `api_request_latency_seconds` — request latency distribution;
- `model_predictions_total` — prediction count by class;
- `model_loaded` — model availability status.

Explain the monitoring flow:

```text
FastAPI → /metrics → Prometheus → Grafana
```

Insert:

- Prometheus target showing API `UP`;
- Prometheus query result;
- Grafana dashboard.

## 13. CI/CD
Describe the GitHub Actions workflow and include the Docker smoke-test result:

```text
lint → train → test → Docker build → artifact upload
```

Insert the successful Actions screenshot.

## 14. Architecture Diagram
Redraw/personalize the architecture from the README and show the complete data-to-monitoring flow.

## 15. Conclusion and Future Work
Discuss limitations and improvements such as:

- model calibration;
- data/model drift monitoring;
- automated retraining;
- authentication and authorization;
- secrets management;
- persistent Prometheus/Grafana storage;
- cloud deployment.

## 16. Assignment Evidence Checklist
Confirm that the report includes screenshots for MLflow, API `/health`, `/predict`, `/metrics`, Docker services, Kubernetes replicas, LoadBalancer service, Prometheus target/query, Grafana dashboard, and GitHub Actions.

## 17. Repository and Video
- GitHub repository:
- API access instructions:
- Demo/video link:
