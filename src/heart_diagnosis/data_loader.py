"""Load and describe the UCI Heart Disease dataset.

Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
We use the classic "processed.cleveland.data" file (303 patients, 14 columns).
The file is cached locally under ``data/`` so the system works offline after
the first download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Order of columns in processed.cleveland.data
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",  # target: 0 = no disease, 1-4 = increasing severity
]

CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


@dataclass
class FeatureInfo:
    """Human-readable metadata for a single dataset feature."""

    name: str
    label: str
    description: str
    # maps a raw numeric code to a clinical phrase; if None the value is numeric
    value_map: Optional[dict] = None
    unit: str = ""

    def render(self, value) -> str:
        """Render a raw value as a clinically meaningful phrase."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return f"{self.label}: unknown"
        if self.value_map is not None:
            key = int(value) if isinstance(value, (int, float)) else value
            phrase = self.value_map.get(key, f"code {value}")
            return f"{self.label}: {phrase}"
        unit = f" {self.unit}" if self.unit else ""
        # show integers without trailing .0
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"{self.label}: {value}{unit}"


# Clinical metadata for every predictor feature.
FEATURES: dict[str, FeatureInfo] = {
    "age": FeatureInfo("age", "Age", "Age of the patient", unit="years"),
    "sex": FeatureInfo(
        "sex", "Sex", "Biological sex", value_map={0: "female", 1: "male"}
    ),
    "cp": FeatureInfo(
        "cp",
        "Chest pain type",
        "Type of chest pain reported",
        value_map={
            1: "typical angina",
            2: "atypical angina",
            3: "non-anginal pain",
            4: "asymptomatic",
        },
    ),
    "trestbps": FeatureInfo(
        "trestbps", "Resting blood pressure", "Resting BP on admission", unit="mm Hg"
    ),
    "chol": FeatureInfo(
        "chol", "Serum cholesterol", "Serum cholesterol", unit="mg/dl"
    ),
    "fbs": FeatureInfo(
        "fbs",
        "Fasting blood sugar > 120 mg/dl",
        "Fasting blood sugar greater than 120 mg/dl",
        value_map={0: "no", 1: "yes"},
    ),
    "restecg": FeatureInfo(
        "restecg",
        "Resting ECG",
        "Resting electrocardiographic result",
        value_map={
            0: "normal",
            1: "ST-T wave abnormality",
            2: "probable/definite left ventricular hypertrophy",
        },
    ),
    "thalach": FeatureInfo(
        "thalach", "Max heart rate achieved", "Maximum heart rate during exercise",
        unit="bpm",
    ),
    "exang": FeatureInfo(
        "exang",
        "Exercise-induced angina",
        "Angina induced by exercise",
        value_map={0: "no", 1: "yes"},
    ),
    "oldpeak": FeatureInfo(
        "oldpeak",
        "ST depression (oldpeak)",
        "ST depression induced by exercise relative to rest",
    ),
    "slope": FeatureInfo(
        "slope",
        "Slope of peak exercise ST segment",
        "Slope of the ST segment during peak exercise",
        value_map={1: "upsloping", 2: "flat", 3: "downsloping"},
    ),
    "ca": FeatureInfo(
        "ca",
        "Major vessels colored by fluoroscopy",
        "Number of major vessels (0-3) colored by fluoroscopy",
    ),
    "thal": FeatureInfo(
        "thal",
        "Thalassemia / perfusion scan",
        "Thallium stress test result",
        value_map={3: "normal", 6: "fixed defect", 7: "reversible defect"},
    ),
}

PREDICTOR_COLUMNS = list(FEATURES.keys())


def _default_cache_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return os.path.join(root, "data", "processed.cleveland.data")


def download_dataset(cache_path: Optional[str] = None, force: bool = False) -> str:
    """Download the Cleveland dataset to ``cache_path`` if not already present."""
    cache_path = cache_path or _default_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if force or not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
        import urllib.request

        urllib.request.urlretrieve(CLEVELAND_URL, cache_path)
    return cache_path


def load_dataset(cache_path: Optional[str] = None) -> pd.DataFrame:
    """Load the dataset as a cleaned DataFrame.

    Adds a binary ``target`` column (1 = heart disease present, 0 = absent).
    Missing values (encoded as ``?``) become ``NaN``.
    """
    cache_path = cache_path or _default_cache_path()
    if not os.path.exists(cache_path):
        download_dataset(cache_path)

    df = pd.read_csv(cache_path, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    return df


def patient_to_dict(row: pd.Series) -> dict:
    """Convert a dataframe row into a plain feature dict (predictors only)."""
    out = {}
    for col in PREDICTOR_COLUMNS:
        val = row[col]
        if pd.isna(val):
            out[col] = None
        else:
            out[col] = float(val)
    return out
