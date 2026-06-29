"""Tests unitaires pour les outils médicaux et le state du graphe."""
import pytest

from src.state import AgentState
from src.tools import calculate_bmi, estimate_drug_dosage


class TestCalculateBMI:
    """Tests pour l'outil de calcul d'IMC."""

    def test_normal_bmi(self):
        result = calculate_bmi.invoke({"weight_kg": 70.0, "height_cm": 175.0})
        assert "25.7" in result or "IMC" in result
        assert "Surpoids" in result or "normale" in result

    def test_underweight(self):
        result = calculate_bmi.invoke({"weight_kg": 45.0, "height_cm": 170.0})
        assert "Insuffisance" in result or "Dénutrition" in result

    def test_obese(self):
        result = calculate_bmi.invoke({"weight_kg": 120.0, "height_cm": 170.0})
        assert "Obésité" in result

    def test_invalid_values(self):
        result = calculate_bmi.invoke({"weight_kg": -5.0, "height_cm": 170.0})
        assert "Erreur" in result

    def test_zero_height(self):
        result = calculate_bmi.invoke({"weight_kg": 70.0, "height_cm": 0.0})
        assert "Erreur" in result


class TestEstimateDrugDosage:
    """Tests pour l'outil d'estimation de posologie."""

    def test_paracetamol_child(self):
        result = estimate_drug_dosage.invoke({
            "drug_name": "paracétamol",
            "weight_kg": 20.0,
            "dose_per_kg": 15.0,
            "frequency_per_day": 4,
        })
        # 20 kg × 15 mg/kg = 300 mg/jour → 75 mg/prise
        assert "300" in result
        assert "75" in result
        assert "paracétamol" in result.lower()

    def test_warning_present(self):
        result = estimate_drug_dosage.invoke({
            "drug_name": "ibuprofène",
            "weight_kg": 70.0,
            "dose_per_kg": 10.0,
            "frequency_per_day": 3,
        })
        assert "AVERTISSEMENT" in result or "médecin" in result.lower()

    def test_invalid_weight(self):
        result = estimate_drug_dosage.invoke({
            "drug_name": "aspirine",
            "weight_kg": 0.0,
            "dose_per_kg": 10.0,
            "frequency_per_day": 2,
        })
        assert "Erreur" in result


class TestAgentState:
    """Tests sur la structure du state LangGraph."""

    def test_state_keys(self):
        required_keys = {
            "messages", "llm_calls", "rewrite_count",
            "retrieved_docs", "last_tool_names", "docs_relevant", "patient_context"
        }
        state_annotations = AgentState.__annotations__
        assert required_keys.issubset(set(state_annotations.keys()))
