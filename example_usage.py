"""
Example usage of Conformal Prediction Coverage Guarantee Skill.
"""

from client import ConformalPredictor


def main():
    print("=== Conformal Prediction Coverage Guarantee Demonstration ===")

    # Simulated holdout calibration data: 5 classes (routes: code, math, search, summarize, chat)
    calibration_labels = [
        "code", "math", "search", "summarize", "chat",
        "code", "math", "search", "code", "summarize"
    ]
    calibration_probs = [
        {"code": 0.85, "math": 0.10, "search": 0.05},
        {"math": 0.70, "code": 0.20, "chat": 0.10},
        {"search": 0.90, "summarize": 0.10},
        {"summarize": 0.65, "chat": 0.25, "code": 0.10},
        {"chat": 0.80, "summarize": 0.20},
        {"code": 0.92, "math": 0.08},
        {"math": 0.45, "code": 0.40, "chat": 0.15},
        {"search": 0.88, "code": 0.12},
        {"code": 0.60, "math": 0.30, "chat": 0.10},
        {"summarize": 0.75, "chat": 0.25}
    ]

    # Target 90% coverage guarantee (alpha = 0.10)
    predictor = ConformalPredictor(alpha=0.10)
    quantile = predictor.calibrate(calibration_labels, calibration_probs)
    print("Calibrated Quantile Cutoff:", round(quantile, 4))

    # Test query with high ambiguity
    ambiguous_query_probs = {
        "code": 0.42,
        "math": 0.38,
        "search": 0.15,
        "chat": 0.05
    }

    result = predictor.predict_set(ambiguous_query_probs)
    print("Ambiguous Query Prediction Set:", result["prediction_set"])
    print("Set Size:", result["set_size"])
    print("Guaranteed Coverage:", f"{result['guaranteed_coverage'] * 100:.1f}%")

    # Evaluate test split
    test_labels = ["code", "math", "search"]
    test_probs = [
        {"code": 0.88, "math": 0.12},
        {"math": 0.55, "code": 0.40, "chat": 0.05},
        {"search": 0.95, "code": 0.05}
    ]
    eval_res = predictor.evaluate_coverage(test_labels, test_probs)
    print("Empirical Test Coverage:", f"{eval_res['empirical_coverage'] * 100:.1f}%")
    print("Average Set Size:", round(eval_res['average_set_size'], 2))


if __name__ == "__main__":
    main()
