FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex code",
    "cp": "Chest pain type code",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dl indicator",
    "restecg": "Resting electrocardiographic result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina indicator",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Number of major vessels",
    "thal": "Thalassemia code",
}
