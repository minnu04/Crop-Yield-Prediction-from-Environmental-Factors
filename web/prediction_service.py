from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_FIELDS = [
    "N",
    "P",
    "K",
    "pH",
    "organic_matter",
    "rainfall",
    "temp_min",
    "temp_max",
    "fertilizer_usage",
    "crop_type",
]

NUMERIC_FIELDS = [
    "N",
    "P",
    "K",
    "pH",
    "organic_matter",
    "rainfall",
    "temp_min",
    "temp_max",
    "fertilizer_usage",
]

CROP_TYPES = ["rice", "maize", "wheat", "barley", "oats"]

FIELD_PROFILES = {
    "N": {"label": "Nitrogen", "ideal": (45.0, 85.0), "unit": "kg/ha"},
    "P": {"label": "Phosphorus", "ideal": (20.0, 60.0), "unit": "kg/ha"},
    "K": {"label": "Potassium", "ideal": (20.0, 60.0), "unit": "kg/ha"},
    "pH": {"label": "pH", "ideal": (6.0, 7.3), "unit": ""},
    "organic_matter": {"label": "Organic matter", "ideal": (2.0, 4.5), "unit": "%"},
    "rainfall": {"label": "Rainfall", "ideal": (120.0, 260.0), "unit": "mm"},
    "temp_min": {"label": "Minimum temperature", "ideal": (15.0, 24.0), "unit": "°C"},
    "temp_max": {"label": "Maximum temperature", "ideal": (26.0, 35.0), "unit": "°C"},
    "fertilizer_usage": {"label": "Fertilizer usage", "ideal": (70.0, 160.0), "unit": "kg/ha"},
}

CROP_PROFILES = {
    "rice": {
        "rainfall": (180.0, 320.0),
        "temp_min": (18.0, 24.0),
        "temp_max": (28.0, 36.0),
        "pH": (5.8, 7.2),
        "note": "Best suited to wetter, warmer conditions.",
    },
    "maize": {
        "rainfall": (110.0, 220.0),
        "temp_min": (16.0, 23.0),
        "temp_max": (26.0, 34.0),
        "pH": (5.8, 7.2),
        "note": "Balanced option for moderate moisture and warm conditions.",
    },
    "wheat": {
        "rainfall": (90.0, 170.0),
        "temp_min": (10.0, 18.0),
        "temp_max": (20.0, 28.0),
        "pH": (6.0, 7.5),
        "note": "Performs well in cooler, moderately dry environments.",
    },
    "barley": {
        "rainfall": (80.0, 150.0),
        "temp_min": (9.0, 17.0),
        "temp_max": (18.0, 26.0),
        "pH": (6.0, 8.0),
        "note": "More tolerant of drier, cooler conditions.",
    },
    "oats": {
        "rainfall": (100.0, 190.0),
        "temp_min": (10.0, 19.0),
        "temp_max": (20.0, 29.0),
        "pH": (5.5, 7.0),
        "note": "A flexible choice for cool, moist environments.",
    },
}


class PredictionInputError(ValueError):
    """Raised when a prediction payload is incomplete or invalid."""


@lru_cache(maxsize=1)
def resolve_project_dir() -> Path:
    web_dir = Path(__file__).resolve().parent
    if (web_dir.parent / "crop-yield-prediction" / "src").exists():
        return web_dir.parent / "crop-yield-prediction"
    return web_dir.parent


PROJECT_DIR = resolve_project_dir()
DATA_PATH = PROJECT_DIR / "data" / "crop_yield_data.csv"


def normalize_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload is None:
        raise PredictionInputError("Request payload is empty.")

    normalized: Dict[str, Any] = {}
    missing: List[str] = []
    invalid_numbers: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in payload or payload[field] in (None, ""):
            missing.append(field)
            continue

        if field in NUMERIC_FIELDS:
            try:
                normalized[field] = float(payload[field])
            except (TypeError, ValueError):
                invalid_numbers.append(field)
        else:
            normalized[field] = str(payload[field]).strip().lower()

    if missing:
        raise PredictionInputError(f"Missing required fields: {', '.join(missing)}")
    if invalid_numbers:
        raise PredictionInputError(f"Invalid numeric values for: {', '.join(invalid_numbers)}")

    crop_type = normalized["crop_type"]
    if crop_type not in CROP_TYPES:
        raise PredictionInputError(f"Invalid crop type: {crop_type}")

    return normalized


@lru_cache(maxsize=1)
def yield_quantiles() -> Dict[str, float]:
    default_quantiles = {"q25": 119.0, "q50": 132.0, "q75": 148.0}
    if not DATA_PATH.exists():
        return default_quantiles

    try:
        df = pd.read_csv(DATA_PATH)
        if "crop_yield" not in df.columns:
            return default_quantiles

        series = pd.to_numeric(df["crop_yield"], errors="coerce").dropna()
        if series.empty:
            return default_quantiles

        quantiles = series.quantile([0.25, 0.5, 0.75]).to_dict()
        return {
            "q25": float(quantiles.get(0.25, default_quantiles["q25"])),
            "q50": float(quantiles.get(0.5, default_quantiles["q50"])),
            "q75": float(quantiles.get(0.75, default_quantiles["q75"])),
        }
    except Exception:
        return default_quantiles


def range_score(value: float, lower: float, upper: float) -> float:
    if lower >= upper:
        return 0.0

    center = (lower + upper) / 2.0
    half_span = max((upper - lower) / 2.0, 1e-6)

    if value < lower:
        return -min(1.0, (lower - value) / max(half_span, 1.0))
    if value > upper:
        return -min(1.0, (value - upper) / max(half_span, 1.0))

    return 1.0 - abs(value - center) / half_span


def describe_range(field: str, value: float) -> str:
    profile = FIELD_PROFILES[field]
    low, high = profile["ideal"]
    label = profile["label"]
    unit = profile["unit"]

    unit_suffix = f" {unit}" if unit else ""
    if value < low:
        return f"{label} is below the preferred range ({low:g}-{high:g}{unit_suffix})."
    if value > high:
        return f"{label} is above the preferred range ({low:g}-{high:g}{unit_suffix})."
    return f"{label} is within the preferred range ({low:g}-{high:g}{unit_suffix})."


def tree_prediction_distribution(model, sample_frame: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "named_steps"):
        raw_prediction = float(model.predict(sample_frame)[0])
        return np.array([raw_prediction], dtype=float)

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    transformed = preprocessor.transform(sample_frame)

    if hasattr(regressor, "estimators_"):
        return np.array([tree.predict(transformed)[0] for tree in regressor.estimators_], dtype=float)

    return np.array([float(regressor.predict(transformed)[0])], dtype=float)


def predict_value(model, sample: Dict[str, Any]) -> float:
    frame = pd.DataFrame([sample])
    return float(model.predict(frame)[0])


def field_importance_map(model) -> Dict[str, float]:
    if not hasattr(model, "named_steps"):
        return {field: 1.0 / len(REQUIRED_FIELDS) for field in REQUIRED_FIELDS}

    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    if not hasattr(regressor, "feature_importances_"):
        return {field: 1.0 / len(REQUIRED_FIELDS) for field in REQUIRED_FIELDS}

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []

    importances = list(regressor.feature_importances_)
    combined: Dict[str, float] = {field: 0.0 for field in REQUIRED_FIELDS}

    for feature_name, importance in zip(feature_names, importances):
        raw_name = feature_name.split("__", 1)[-1]
        if raw_name.startswith("crop_type_"):
            combined["crop_type"] += float(importance)
        else:
            combined[raw_name] = combined.get(raw_name, 0.0) + float(importance)

    total = sum(combined.values()) or 1.0
    return {field: value / total for field, value in combined.items()}


def feature_contributions(model, sample: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    importances = field_importance_map(model)
    scored: List[Dict[str, Any]] = []

    for field in NUMERIC_FIELDS:
        value = float(sample[field])
        profile = FIELD_PROFILES[field]
        lower, upper = profile["ideal"]
        alignment = range_score(value, lower, upper)
        importance = importances.get(field, 0.0)
        impact = round(alignment * importance * 100, 2)
        if alignment < 0:
            impact = round(-abs(impact), 2)

        scored.append(
            {
                "feature": field,
                "label": profile["label"],
                "value": round(value, 2),
                "impact": impact,
                "message": describe_range(field, value),
            }
        )

    scored.sort(key=lambda item: item["impact"], reverse=True)
    top_positive = [item for item in scored if item["impact"] > 0][:3]
    top_negative = sorted([item for item in scored if item["impact"] <= 0], key=lambda item: item["impact"])[:3]

    if top_positive and top_negative:
        summary = (
            f"{top_positive[0]['label']} is supporting yield, while {top_negative[0]['label'].lower()} needs the most attention."
        )
    elif top_positive:
        summary = f"{top_positive[0]['label']} is contributing most positively to the forecast."
    elif top_negative:
        summary = f"{top_negative[0]['label']} is the main limiting factor in this scenario."
    else:
        summary = "The model sees a balanced input profile with no strong limiting factor."

    driver_breakdown = sorted(scored, key=lambda item: abs(item["impact"]), reverse=True)
    return top_positive, top_negative, summary, driver_breakdown


def explanation_highlights(
    prediction: float,
    confidence: Dict[str, float],
    risk: Dict[str, Any],
    top_positive: List[Dict[str, Any]],
    top_negative: List[Dict[str, Any]],
) -> Dict[str, Any]:
    quantiles = yield_quantiles()
    highlights: List[str] = []
    headline = "The model sees a balanced input profile."

    if prediction >= quantiles["q75"]:
        headline = f"Yield is tracking above the historical upper quartile at {prediction:.2f} units."
    elif prediction <= quantiles["q25"]:
        headline = f"Yield is sitting in the lower quartile at {prediction:.2f} units."
    else:
        headline = f"Yield is near the middle of the historical range at {prediction:.2f} units."

    highlights.append(headline)

    confidence_score = confidence.get("score", 0.0)
    if confidence_score >= 0.75:
        highlights.append("Confidence is strong, so the prediction band is relatively stable.")
    elif confidence_score >= 0.5:
        highlights.append("Confidence is moderate, which means the estimate is useful but not exact.")
    else:
        highlights.append("Confidence is limited, so treat the estimate as directional guidance.")

    risk_level = risk.get("level", "low")
    if risk_level == "high":
        highlights.append("Risk is high, so input corrections should be prioritized before planting.")
    elif risk_level == "medium":
        highlights.append("Risk is moderate, with a few inputs needing attention.")
    else:
        highlights.append("Risk is low, so current conditions are broadly supportive.")

    if top_positive and top_negative:
        highlights.append(
            f"{top_positive[0]['label']} is helping most, while {top_negative[0]['label']} is the main constraint."
        )
    elif top_positive:
        highlights.append(f"{top_positive[0]['label']} is the strongest positive driver in this scenario.")
    elif top_negative:
        highlights.append(f"{top_negative[0]['label']} needs the most attention to improve yield.")

    return {
        "headline": headline,
        "highlights": highlights[:4],
    }


def confidence_band(model, sample: Dict[str, Any]) -> Dict[str, float]:
    frame = pd.DataFrame([sample])
    preds = tree_prediction_distribution(model, frame)
    prediction = float(np.mean(preds))

    if len(preds) > 1:
        lower, upper = np.percentile(preds, [10, 90]).astype(float)
    else:
        lower, upper = prediction * 0.9, prediction * 1.1

    width = max(upper - lower, 1e-6)
    score = float(np.clip(1.0 - (width / max(abs(prediction), 1.0)), 0.05, 0.99))
    return {
        "prediction": prediction,
        "lower": float(round(lower, 2)),
        "upper": float(round(upper, 2)),
        "score": float(round(score, 2)),
        "width": float(round(width, 2)),
    }


def stress_signals(sample: Dict[str, Any]) -> List[str]:
    signals: List[str] = []

    if sample["rainfall"] < 120:
        signals.append("Rainfall is low for strong crop development.")
    if sample["temp_min"] < 14:
        signals.append("Minimum temperature is below the comfort range for this crop.")
    if sample["temp_max"] > 36:
        signals.append("Maximum temperature is high and may create heat stress.")
    if sample["pH"] < 5.8 or sample["pH"] > 7.5:
        signals.append("Soil pH is outside the optimal nutrient-uptake range.")
    if min(sample["N"], sample["P"], sample["K"]) < 25:
        signals.append("One or more primary nutrients look under-supplied.")
    if sample["fertilizer_usage"] < 70:
        signals.append("Fertilizer usage is low and may limit yield potential.")

    return signals


def risk_assessment(model, sample: Dict[str, Any], prediction: float, confidence: Dict[str, float]) -> Dict[str, Any]:
    quantiles = yield_quantiles()
    reasons: List[str] = []
    severity = 0

    if prediction <= quantiles["q25"]:
        severity += 2
        reasons.append("Predicted yield is in the lower quartile of historical performance.")
    elif prediction >= quantiles["q75"]:
        severity -= 1

    if confidence["width"] / max(prediction, 1.0) > 0.35:
        severity += 2
        reasons.append("Confidence band is wide, so forecast uncertainty is elevated.")
    elif confidence["width"] / max(prediction, 1.0) > 0.2:
        severity += 1
        reasons.append("Confidence band shows moderate uncertainty.")

    stress = stress_signals(sample)
    reasons.extend(stress)
    severity += min(len(stress), 3)

    if severity >= 4:
        level = "high"
    elif severity >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "badge": level.upper(),
        "reasons": reasons[:4] if reasons else ["No major agronomic stress detected."],
        "stressFactors": stress[:4],
    }


def advisory_messages(sample: Dict[str, Any], risk: Dict[str, Any], negative_factors: List[Dict[str, Any]]) -> List[str]:
    advice: List[str] = []
    stress = set(risk.get("reasons", []))

    if any("Rainfall" in reason for reason in stress):
        advice.append("Consider irrigation support or moisture conservation practices.")
    if any("pH" in reason for reason in stress):
        advice.append("Adjust soil pH using lime or organic amendments after a soil test.")
    if any("nutrients" in reason.lower() for reason in stress):
        advice.append("Balance N, P, and K inputs according to a soil test recommendation.")
    if any("temperature" in reason.lower() for reason in stress):
        advice.append("Shift sowing dates or choose a crop with better temperature tolerance.")

    if not advice and negative_factors:
        top_issue = negative_factors[0]["feature"]
        if top_issue == "rainfall":
            advice.append("Increase irrigation or choose a moisture-tolerant crop variety.")
        elif top_issue in {"N", "P", "K"}:
            advice.append("Correct the nutrient balance before sowing to unlock higher yield potential.")
        elif top_issue == "pH":
            advice.append("Improve soil pH balance with field-specific soil amendments.")
        else:
            advice.append("Fine-tune the limiting input before finalizing the crop plan.")

    if not advice:
        advice.append("Conditions look balanced. Maintain current agronomic practices and monitor weather.")

    return advice[:4]


def crop_suitability_score(crop: str, sample: Dict[str, Any]) -> Tuple[float, str]:
    profile = CROP_PROFILES[crop]
    rainfall_score = range_score(sample["rainfall"], *profile["rainfall"])
    temp_min_score = range_score(sample["temp_min"], *profile["temp_min"])
    temp_max_score = range_score(sample["temp_max"], *profile["temp_max"])
    pH_score = range_score(sample["pH"], *profile["pH"])

    combined = (rainfall_score + temp_min_score + temp_max_score + pH_score) / 4.0
    if combined >= 0.6:
        reason = profile["note"]
    elif combined >= 0.2:
        reason = "Conditions are moderately aligned with this crop profile."
    else:
        reason = "This crop is less aligned with the current weather and soil profile."

    return combined, reason


def crop_recommendations(model, sample: Dict[str, Any]) -> Dict[str, Any]:
    ranked: List[Dict[str, Any]] = []

    for crop in CROP_TYPES:
        candidate = dict(sample)
        candidate["crop_type"] = crop
        predicted = predict_value(model, candidate)
        suitability, reason = crop_suitability_score(crop, candidate)
        ranked.append(
            {
                "crop": crop,
                "expectedYield": float(round(predicted, 2)),
                "suitabilityScore": float(round(max(suitability, 0.0), 2)),
                "reason": reason,
            }
        )

    ranked.sort(key=lambda item: item["expectedYield"], reverse=True)
    best = ranked[0]
    selected = next((item for item in ranked if item["crop"] == sample["crop_type"]), None)
    return {
        "bestCrop": best["crop"],
        "selectedCrop": sample["crop_type"],
        "selectedCropYield": selected["expectedYield"] if selected else None,
        "bestReason": best["reason"],
        "alternatives": ranked,
        "topAlternatives": ranked[:3],
    }


def build_prediction_response(model, payload: Dict[str, Any]) -> Dict[str, Any]:
    sample = normalize_input(payload)
    confidence = confidence_band(model, sample)
    prediction = float(round(confidence["prediction"], 2))
    top_positive, top_negative, summary, driver_breakdown = feature_contributions(model, sample)
    risk = risk_assessment(model, sample, prediction, confidence)
    recommendation = crop_recommendations(model, sample)
    advisory = advisory_messages(sample, risk, top_negative)
    narrative = explanation_highlights(prediction, confidence, risk, top_positive, top_negative)

    selected_crop = sample["crop_type"]
    best_crop = recommendation["bestCrop"]
    if selected_crop != best_crop:
        recommendation["selectionNote"] = (
            f"Your selected crop ({selected_crop}) is not the top performer. {best_crop} currently has the strongest forecast."
        )
    else:
        recommendation["selectionNote"] = f"Your selected crop ({selected_crop}) is currently the best predicted option."

    return {
        "predictedYield": prediction,
        "confidence": {
            "lower": confidence["lower"],
            "upper": confidence["upper"],
            "score": confidence["score"],
        },
        "risk": risk,
        "explanation": {
            "topPositive": top_positive,
            "topNegative": top_negative,
            "summary": summary,
            "narrative": narrative,
            "driverBreakdown": driver_breakdown,
        },
        "recommendation": recommendation,
        "advisory": advisory,
        "featureContributions": top_positive + top_negative,
        "input": sample,
    }


def apply_scenario_adjustments(sample: Dict[str, Any], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    adjusted = dict(sample)

    if "rainfall" in adjustments and adjustments["rainfall"] not in (None, ""):
        adjusted["rainfall"] = float(adjustments["rainfall"])

    if "fertilizer_usage" in adjustments and adjustments["fertilizer_usage"] not in (None, ""):
        adjusted["fertilizer_usage"] = float(adjustments["fertilizer_usage"])

    temp_delta = float(adjustments.get("temp_delta", 0.0) or 0.0)
    if temp_delta:
        adjusted["temp_min"] = float(adjusted["temp_min"] + temp_delta)
        adjusted["temp_max"] = float(adjusted["temp_max"] + temp_delta)

    return adjusted


def build_simulation_response(model, payload: Dict[str, Any], adjustments: Dict[str, Any]) -> Dict[str, Any]:
    baseline_input = normalize_input(payload)
    scenario_input = apply_scenario_adjustments(baseline_input, adjustments or {})

    baseline = build_prediction_response(model, baseline_input)
    scenario = build_prediction_response(model, scenario_input)

    yield_delta = round(scenario["predictedYield"] - baseline["predictedYield"], 2)
    confidence_delta = round(scenario["confidence"]["score"] - baseline["confidence"]["score"], 2)
    risk_order = {"low": 1, "medium": 2, "high": 3}
    baseline_risk = baseline["risk"]["level"]
    scenario_risk = scenario["risk"]["level"]
    risk_shift = risk_order.get(scenario_risk, 1) - risk_order.get(baseline_risk, 1)

    return {
        "baseline": baseline,
        "scenario": scenario,
        "delta": {
            "yield": yield_delta,
            "confidence": confidence_delta,
            "riskShift": risk_shift,
            "riskLabel": "improved" if risk_shift < 0 else "worsened" if risk_shift > 0 else "unchanged",
        },
        "adjustments": {
            "rainfall": scenario_input["rainfall"],
            "fertilizer_usage": scenario_input["fertilizer_usage"],
            "temp_delta": round(float(adjustments.get("temp_delta", 0.0) or 0.0), 2),
        },
    }
