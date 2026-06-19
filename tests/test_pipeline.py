"""End-to-end tests for the crew, protocols, intake and data loading."""

from heart_diagnosis.crew import HeartDiagnosisCrew
from heart_diagnosis.data_loader import PREDICTOR_COLUMNS, load_dataset, patient_to_dict
from heart_diagnosis.intake import parse_complaint


def test_dataset_loads():
    df = load_dataset()
    assert len(df) > 250
    assert "target" in df.columns
    assert set(df["target"].unique()) <= {0, 1}


def test_crew_diagnose_high_risk():
    crew = HeartDiagnosisCrew(engine="heuristic")
    patient = {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
        "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
        "ca": 3, "thal": 7,
    }
    result = crew.diagnose(patient)
    assert result.final.final_decision == "Positive"
    assert result.critic is not None
    assert result.risk is not None


def test_crew_insufficient_data():
    crew = HeartDiagnosisCrew(engine="heuristic")
    empty = {c: None for c in PREDICTOR_COLUMNS}
    result = crew.diagnose(empty, complaint="my tummy hurts")
    assert result.final.final_decision == "Negative"
    assert "Insufficient" in result.final.explanation


def test_debate_protocol_runs_two_rounds():
    crew = HeartDiagnosisCrew(engine="heuristic", protocol="debate")
    df = load_dataset()
    patient = patient_to_dict(df.iloc[1])
    result = crew.diagnose(patient)
    assert any("Round 1" in n for n in result.notes)


def test_intake_heuristic_extracts_features():
    feats = parse_complaint(
        "58 year old man with chest pain when climbing stairs, cholesterol 290",
        engine="heuristic",
    )
    assert feats["age"] == 58
    assert feats["sex"] == 1
    assert feats["chol"] == 290
