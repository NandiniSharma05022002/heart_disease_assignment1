# Heart Disease MLOps — End-to-End Assignment

This repository implements the complete MLOps assignment for binary heart-disease classification using the UCI Cleveland Heart Disease dataset.

**End-to-end flow:**

```text
UCI data
  → cleaning + EDA
  → reproducible sklearn preprocessing pipeline
  → Logistic Regression + Random Forest
  → 5-fold CV + holdout evaluation
  → MLflow experiment tracking
  → final reusable pickle artifact
  → FastAPI /predict + /health + /metrics
  → Docker
  → Kubernetes/Minikube with LoadBalancer
  → Prometheus
  → Grafana dashboard
  → Pytest + Ruff + GitHub Actions
  → final report + screenshots + demo video
```

The implementation is intentionally local-first so every assignment component can be demonstrated without a paid cloud account. The Kubernetes deployment uses Minikube, which is explicitly permitted by the assignment.

## 1. Assignment coverage

The assignment requires data acquisition/EDA, two classifiers and evaluation, MLflow tracking, reusable packaging, tests and CI/CD, Docker, Kubernetes LoadBalancer deployment, monitoring/logging, and a professional report. fileciteturn6file0L10-L40

The repository contains the requested code, Dockerfile, requirements, cleaned dataset/download path, EDA/training/inference scripts, tests, GitHub Actions workflow, Kubernetes manifests, screenshot directory, and report template. fileciteturn6file0L41-L56

## 2. Prerequisites

- Python 3.11
- Docker Desktop
- `kubectl`
- Minikube
- Git

Optional for local MLflow UI:

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --host 127.0.0.1 --port 5000
```

## 3. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the active environment:

```bash
which python
python --version
python -m pip --version
```

## 4. Data acquisition, cleaning and EDA

If `data/processed/heart_disease.csv` is not already present:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

Generate the required EDA visuals:

```bash
python scripts/eda.py
```

Generated figures:

- `reports/figures/feature_distributions.png`
- `reports/figures/class_balance.png`
- `reports/figures/correlation_heatmap.png`

The preprocessing step converts the original UCI target valueminikube service heart-disease-api -n heart-disease --urls `0..4` to binary `0/1` and preserves missing values for the sklearn pipeline to handle. The model pipeline uses median imputation + standardization for numeric features and most-frequent imputation + one-hot encoding for categorical features.

## 5. Model training + MLflow

Train both candidate classifiers:

```bash
python -m src.heart_disease_mlops.train
```

The training script:

- uses random seed `42`;
- performs a stratified 80/20 train-test split;
- performs stratified 5-fold cross-validation on the training split;
- evaluates accuracy, precision, recall and ROC-AUC;
- compares Logistic Regression and Random Forest;
- tunes Logistic Regression `C` over `[0.1, 1.0, 10.0]` and Random Forest `min_samples_leaf` over `[1, 2, 4]` with stratified 5-fold CV;
- logs parameters, metrics, plots and model artifacts to MLflow;
- selects both hyperparameters and the final model using mean cross-validation ROC-AUC; the 20% holdout is used only for final reporting;
- saves the **complete fitted sklearn Pipeline** as `artifacts/model.pkl`;
- writes `artifacts/model_metadata.json` and `artifacts/model_comparison.csv`.

### MLflow tracking

MLflow uses a SQLite backend stored in the project root:

```text
mlflow.db
```

Start the UI in another terminal:

```bash
python -m mlflow ui \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`.

Capture one screenshot showing both model runs, their parameters/metrics, and model/plot artifacts.

> The MLflow sklearn flavor may use `skops` serialization. The training code explicitly trusts only the model types required by the locally created Logistic Regression and Random Forest artifacts. The standalone serving artifact is still Python pickle, as required by the selected packaging approach.

## 6. Local reusable-artifact inference

The final artifact is a trusted-source Python pickle containing the full preprocessing + classifier pipeline.

Run the standalone inference script:

```bash
python scripts/inference.py
```

You can also pass your own JSON file:

```bash
python scripts/inference.py --input data/sample_patient.json
```

Python pickle must only be loaded from a trusted source.

## 7. Tests and linting

Run unit tests:

```bash
pytest -q
```

Run linting:

```bash
ruff check .
```

The tests cover:

- data cleaning and target conversion;
- Logistic Regression pipeline fit/predict;
- Random Forest pipeline fit/predict;
- packaged pickle existence/loadability;
- FastAPI `/health`, `/predict`, and `/metrics`.

## 8. FastAPI service and monitoring metrics

Start the API:

```bash
uvicorn src.heart_disease_mlops.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok","model_loaded":true}
```

Swagger UI:

```text
http://localhost:8000/docs
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,
    "fbs":1,"restecg":0,"thalach":150,"exang":0,
    "oldpeak":2.3,"slope":2,"ca":0,"thal":3
  }'
```

Prometheus endpoint:

```bash
curl -s http://localhost:8000/metrics | grep -E 'api_requests_total|api_request_latency_seconds|model_predictions_total|model_loaded'
```

The service exposes these custom monitoring metrics:

| Metric                          | Purpose                                            |
| ------------------------------- | -------------------------------------------------- |
| `api_requests_total`          | Request count by endpoint, method and HTTP status  |
| `api_request_latency_seconds` | Request latency distribution                       |
| `model_predictions_total`     | Prediction count by predicted class                |
| `model_loaded`                | Model availability:`1` loaded, `0` unavailable |

The API also logs method, path, status and latency for every request.

## 9. Docker + Prometheus + Grafana

Build the API image:

```bash
docker build -t heart-disease-api:latest .
```

Run the complete local stack:

```bash
docker compose up --build -d
```

Verify:

```bash
docker compose ps
```

Endpoints:

```text
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```

Grafana is provisioned automatically with the Prometheus datasource and a dashboard containing request count/rate, predictions, status codes, latency and model health.

Generate several prediction requests while the stack is running, then inspect Prometheus and Grafana.

Useful PromQL queries:

```promql
sum(api_requests_total)
```

```promql
sum(rate(api_requests_total[1m]))
```

```promql
sum(model_predictions_total)
```

```promql
model_loaded
```

```promql
rate(api_request_latency_seconds_sum[1m]) / rate(api_request_latency_seconds_count[1m])
```

## 10. Kubernetes / Minikube deployment

The assignment explicitly allows Minikube and requires a Deployment plus LoadBalancer/Ingress exposure. fileciteturn6file0L25-L29

Start Minikube:

```bash
minikube start
```

Build the application image inside Minikube:

Apply all manifests with Kustomize so the namespace is created before namespaced resources:

```bash
minikube kubectl -- apply -k k8s/
```

Verify the API:

```bash
minikube kubectl -- get deployments -n heart-disease
minikube kubectl -- get pods -n heart-disease
minikube kubectl -- get services -n heart-disease
```

The API service is intentionally:

```yaml
type: LoadBalancer
```

Get the Minikube LoadBalancer URL:

```bash
minikube service heart-disease-api -n heart-disease --url
```

if above command gives error please run

```bash
minikube image build -t heart-disease-api:latest .
```

Use the returned URL to test:

```text
<URL>/health
<URL>/docs
<URL>/metrics
```

For `/predict`, use the same JSON payload shown in the local API section.

### Kubernetes readiness/liveness

The API Deployment has:

- 2 replicas;
- readiness probe on `/health`;
- liveness probe on `/health`;
- CPU/memory requests and limits;
- `IfNotPresent` image policy so the locally built Minikube image is used.

## 11. Kubernetes monitoring

The same Kustomize application deploys Prometheus and Grafana.

Verify:

```bash
minikube kubectl -- get pods -n heart-disease
minikube kubectl -- get services -n heart-disease
```

Prometheus scrapes the API through Kubernetes DNS:

```text
http://heart-disease-api:80/metrics
```

Port-forward Prometheus:

```bash
minikube kubectl -- port-forward svc/prometheus -n heart-disease 9090:9090
```

Open:

```text
http://localhost:9090/targets
```

The `heart-disease-api` target must show **UP**.

Grafana stays inside the cluster; forward its port to view the dashboard locally:

```bash
minikube kubectl -- port-forward svc/grafana -n heart-disease 3000:3000
```

Then open `http://localhost:3000`.

Login on a fresh local installation:

```text
username: admin
password: admin
```

The dashboard is automatically provisioned.

## 12. GitHub Actions CI and Kubernetes deployment

Workflow file:

```text
.github/workflows/ci.yml
```

The GitHub Actions workflow performs continuous integration (CI):

```text
checkout
  ↓
install requirements
  ↓
ruff lint
  ↓
model training + MLflow logging
  ↓
pytest + coverage
  ↓
Docker build
  ↓
Docker health/metrics smoke test
  ↓
upload model, MLflow and test artifacts (also when an earlier step fails)
```

Push the repository to GitHub and capture a successful Actions run. The workflow is designed to fail when linting, training, tests or the Docker smoke test fails. GitHub Actions uploads any generated model, MLflow and test artifacts even when a prior step fails; the Actions run also retains its step logs.

Production deployment is performed separately to the local Minikube cluster using the Deployment and LoadBalancer Service manifests in `k8s/` (see Section 10). This local deployment is not triggered by GitHub Actions: a hosted Actions runner cannot reach your local Minikube cluster without an externally reachable cluster and configured credentials. After deploying, verify the API and capture the deployment screenshots listed below.

## 13. Assignment evidence checklist

Capture screenshots from your actual run. The assignment specifically asks for deployment screenshots and a professional report containing setup, EDA/model choices, experiment tracking, architecture, CI/CD/deployment evidence and repository information. fileciteturn6file0L33-L50

Recommended evidence set:

1. EDA feature distributions.
2. Class-balance chart.
3. Correlation heatmap.
4. MLflow experiment showing both model runs.
5. Model comparison metrics / final selected model.
6. Swagger `/docs`.
7. Successful `/health`.
8. Successful `/predict`.
9. `/metrics` showing custom metrics.
10. Docker Compose services running.
11. Kubernetes pods showing `2/2` API replicas ready.
12. Kubernetes Service showing `LoadBalancer`.
13. Successful LoadBalancer `/health` and `/predict`.
14. Prometheus target showing API `UP`.
15. Prometheus query result.
16. Grafana monitoring dashboard.
17. Successful GitHub Actions workflow.

Do not fabricate values or screenshots. Use the exact results produced by your own execution.

## 14. Final report

Use `REPORT_TEMPLATE.md` as the structure for the final 10-page report. The assignment requires a written report and a short video demonstrating the overall pipeline. fileciteturn6file0L41-L52

The report should contain your actual:

- dataset and cleaning decisions;
- EDA observations;
- model configurations and selection rationale;
- cross-validation and holdout metrics;
- MLflow screenshots/run IDs;
- reusable artifact explanation;
- API examples;
- Docker evidence;
- Kubernetes/LoadBalancer evidence;
- Prometheus/Grafana evidence;
- CI/CD screenshot;
- architecture diagram;
- repository link;
- video/demo link.

## 15. Repository structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   ├── model.pkl
│   ├── model_metadata.json
│   └── model_comparison.csv
├── data/
│   ├── raw/
│   └── processed/heart_disease.csv
├── grafana/
│   ├── dashboards/heart-disease-api.json
│   └── provisioning/
├── k8s/
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── prometheus-config.yaml
│   ├── prometheus.yaml
│   └── grafana.yaml
├── monitoring/prometheus.yml
├── notebooks/01_eda_and_modeling.md
├── reports/figures/
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── eda.py
│   └── inference.py
├── src/heart_disease_mlops/
│   ├── api.py
│   ├── pipeline.py
│   ├── schema.py
│   └── train.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── REPORT_TEMPLATE.md
```

## 16. Production-readiness note

The assignment requires scripts to work from a clean requirements-based setup, the model to serve in an isolated Docker environment, and the pipeline to fail clearly on code/test errors. fileciteturn6file0L52-L56

The repository therefore keeps the serving artifact self-contained, tests the artifact and API, and smoke-tests the Docker image in CI.

For a real production system, the next improvements would be authentication, secret management, persistent monitoring storage, model/data drift detection, calibration, automated retraining, and a managed cloud deployment.
