# Heart Disease MLOps Pipeline

This repository contains an end-to-end machine learning pipeline for binary heart-disease classification using the UCI Cleveland Heart Disease dataset.

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
  → project report and demo
```

The project runs locally and uses Minikube for Kubernetes deployment, so it can be explored without a paid cloud account.

## 1. Project overview

The repository includes data preparation and EDA, model training and MLflow tracking, a FastAPI prediction service, Docker packaging, Kubernetes manifests, Prometheus/Grafana monitoring, tests, and GitHub Actions CI.

## 2. Prerequisites

- Python 3.11 (the workflow and Docker image use Python 3.11)
- Docker Engine/Desktop
- Minikube
- Git

On a Linux VM, `kubectl` is optional: use `minikube kubectl -- <kubectl arguments>` in the commands below. Make sure the VM user can run Docker (`docker info` should work without a permission error).

Optional for local MLflow UI:

```bash
python -m mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow.db" --host 127.0.0.1 --port 5000
```

## 3. Environment setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the active environment:

```bash
which python
python --version
python -m pip --version
```

## 4. Data acquisition, cleaning and EDA

The scripts create the required `data/` subdirectories. If the processed file is missing (for example, after cloning without data files), run:

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

The preparation step converts the original UCI target values `0..4` to binary `0/1` and preserves missing feature values for the sklearn pipeline to handle. The model pipeline uses median imputation and standardization for numeric features, and most-frequent imputation and one-hot encoding for categorical features.

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

Always run training and the MLflow UI from the same activated `.venv` using `python -m ...`; this avoids accidentally running a different globally installed MLflow. Do not copy `mlflow.db` or `mlruns/` from another machine into a fresh VM: MLflow databases can contain machine-specific artifact paths and migration versions. If you already copied an old database, stop MLflow and move the old files aside as backups, then run training to create a VM-local database and artifacts. For example, only when these paths exist:

```bash
mv mlflow.db mlflow.db.old
mv mlruns mlruns.old
python -m src.heart_disease_mlops.train
```

Keep the backup if you need the old runs. If `python -m mlflow ui` reports that `index.html` is missing, confirm MLflow is imported from the active `.venv` with `python -c 'import mlflow; print(mlflow.__file__)'`; reinstall the PyPI package into that environment with `python -m pip install --force-reinstall --no-cache-dir 'mlflow>=2.17,<4'`.

Start the UI in another terminal:

```bash
python -m mlflow ui \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`.

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

The Kubernetes manifests define a Deployment and LoadBalancer Service for local Minikube deployment.

Start Minikube:

```bash
minikube start --driver=docker
```

To discard and recreate the cluster instead, run `minikube delete` first. This deletes the Minikube cluster and its Kubernetes resources, including Deployments and Prometheus history, but does not delete the project source files.

Build the image with Docker on the VM, then explicitly load it into Minikube. This uses the Docker build cache on repeat builds; Minikube and the VM's Docker image stores are separate, so the `minikube image load` step is required. Run these commands from the project root, where the `Dockerfile` and `artifacts/model.pkl` exist:

```bash
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
minikube image ls | grep heart-disease-api
```

If you prefer Minikube to perform the build itself, use `minikube image build -t heart-disease-api:latest .` instead of the `docker build` and `minikube image load` commands above. The image tag must match `image:` in `k8s/deployment.yaml` (`heart-disease-api:latest`).

Apply all manifests with Kustomize so the namespace is created before namespaced resources:

```bash
minikube kubectl -- apply -k k8s/
minikube kubectl -- rollout status deployment/heart-disease-api -n heart-disease
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

On a remote VM, the Minikube URL may use a private VM network address that is not reachable directly from your laptop. To verify from a shell on the VM, port-forward the Kubernetes Service in one terminal:

```bash
minikube kubectl -- port-forward svc/heart-disease-api -n heart-disease 8000:80
```

Keep that terminal open, then use another VM terminal to call `http://127.0.0.1:8000`:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @data/sample_patient.json
```

Port-forward Prometheus and Grafana in separate terminals as described in Section 11. Use an SSH local port-forward from your laptop if you need to open these VM-local ports in your laptop browser.

If the API pods show `ImagePullBackOff` or `ErrImageNeverPull`, Kubernetes cannot find or pull the image configured in the Deployment. The LoadBalancer Service will not have a backend pod until at least one API pod is ready. Check the image value and detailed pod events:

```bash
minikube kubectl -- get deployment heart-disease-api -n heart-disease \
  -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
minikube kubectl -- get pods -n heart-disease
minikube kubectl -- describe pod -n heart-disease -l app=heart-disease-api
```

The image name and tag printed from the Deployment must match the image loaded into Minikube. If the Deployment uses `heart-disease-api:latest`, rebuild/import that exact tag, reapply the manifests, and restart the Deployment:

```bash
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
minikube kubectl -- apply -k k8s/
minikube kubectl -- rollout restart deployment/heart-disease-api -n heart-disease
minikube kubectl -- rollout status deployment/heart-disease-api -n heart-disease
minikube kubectl -- get pods -n heart-disease
```

If the pods still cannot start, inspect the **Events** section at the end of `describe pod`. Common causes are a tag mismatch, building from a directory without the project `Dockerfile`, or an image build failure. Do not continue to endpoint checks until both API pods show `Running` and `1/1` ready. Then retry:

```bash
minikube service heart-disease-api -n heart-disease --url
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

The workflow fails if linting, training, tests, image building, or the container smoke test fails. It uploads generated model and test artifacts even if an earlier step fails; GitHub retains the run logs and uploaded artifacts.

Deployment to Minikube is performed separately with the Deployment and LoadBalancer Service manifests in `k8s/` (see Section 10). The workflow does not deploy to the local cluster; a hosted Actions runner cannot reach it without an externally reachable cluster and configured credentials.

## 13. Project report and demo

- [Project report](heart_disease_mlops_report.pdf) — methodology, results, architecture, and implementation details.
- [Demo video](mlops_assignment1.mp4) — walkthrough of the pipeline.

Repository: [heart_disease_assignment1](https://github.com/NandiniSharma05022002/heart_disease_assignment1). The GitHub Actions workflow is in `.github/workflows/ci.yml`; Kubernetes manifests are in `k8s/`.

## 14. Repository structure

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
├── heart_disease_mlops_report.pdf
├── mlops_assignment1.mp4
├── pyproject.toml
└── requirements.txt
```
