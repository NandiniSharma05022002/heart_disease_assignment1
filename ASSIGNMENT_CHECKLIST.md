# Assignment 01 Completion Checklist

Use this as the final submission checklist. Mark an item only after producing your own evidence.

| Requirement | Implementation | Evidence to capture | Status |
|---|---|---|---|
| Data acquisition | `scripts/download_data.py` | raw/downloaded dataset | ☐ |
| Data cleaning | `scripts/prepare_data.py` | processed CSV + missing-value output | ☐ |
| EDA | `scripts/eda.py` | 3 EDA figures | ☐ |
| Feature engineering | `pipeline.py` | preprocessing pipeline code | ☐ |
| Two classifiers | Logistic Regression + Random Forest | model comparison | ☐ |
| CV + metrics | 5-fold stratified CV + holdout | accuracy/precision/recall/ROC-AUC | ☐ |
| Model selection/tuning | logged classifier hyperparameters + metric comparison | MLflow parameters/results | ☐ |
| MLflow tracking | SQLite MLflow backend | MLflow screenshot | ☐ |
| Reusable model | `artifacts/model.pkl` | artifact + metadata | ☐ |
| Reproducibility | `requirements.txt` + fixed seed + full pipeline | clean setup proof | ☐ |
| Unit tests | `tests/` | `pytest` output | ☐ |
| CI/CD | `.github/workflows/ci.yml` | successful Actions run | ☐ |
| Docker | `Dockerfile` + Compose | build/run/smoke test | ☐ |
| Kubernetes | Minikube manifests | pods/deployment output | ☐ |
| LoadBalancer | `k8s/service.yaml` | Service type + URL | ☐ |
| API | FastAPI `/predict` | successful JSON prediction | ☐ |
| API monitoring | `/metrics` + request logging | metrics output/logs | ☐ |
| Prometheus | Kubernetes + Docker scrape config | target `UP` | ☐ |
| Grafana | provisioned dashboard | dashboard screenshot | ☐ |
| Report | `REPORT_TEMPLATE.md` | final 10-page PDF/DOCX | ☐ |
| Architecture | README/report diagram | architecture screenshot/figure | ☐ |
| Repository | GitHub | repository link | ☐ |
| Video | pipeline walkthrough | video link/file | ☐ |
