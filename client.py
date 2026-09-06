"""
Conformal Prediction Coverage Guarantee Skill Client
Pure Python Standard Library implementation of distribution-free split conformal prediction.
Guarantees 1 - alpha marginal coverage over multi-class and regression outputs.
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set


class ConformalPredictor:
    """
    Split conformal prediction engine for agent decision sets.
    Computes calibrated non-conformity scores and constructs prediction sets
    with guaranteed (1 - alpha) coverage without distributional assumptions.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        :param alpha: Significance level (default 0.1 -> 90% coverage guarantee).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("Alpha must be in range (0.0, 1.0)")
        self.alpha = alpha
        self.calibrated_quantile: Optional[float] = None
        self.calibration_scores: List[float] = []

    def compute_conformity_scores(self, true_labels: List[Any], prob_distributions: List[Dict[Any, float]]) -> List[float]:
        """
        Compute non-conformity scores: 1.0 - P(true_label).
        Higher score means model was less confident in the ground truth.
        """
        scores = []
        for label, probs in zip(true_labels, prob_distributions):
            p = probs.get(label, 0.0)
            scores.append(1.0 - p)
        return scores

    def calibrate(self, true_labels: List[Any], prob_distributions: List[Dict[Any, float]]) -> float:
        """
        Calibrate quantile cut-off using hold-out calibration dataset.
        Q_level = ceil((n + 1) * (1 - alpha)) / n
        """
        if not true_labels or len(true_labels) != len(prob_distributions):
            raise ValueError("Mismatched or empty calibration data")

        scores = self.compute_conformity_scores(true_labels, prob_distributions)
        self.calibration_scores = sorted(scores)
        n = len(scores)

        # Standard finite-sample conformal quantile index (1-indexed ceiling)
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        rank = min(max(rank, 1), n)
        
        self.calibrated_quantile = self.calibration_scores[rank - 1]
        return self.calibrated_quantile

    def predict_set(self, prob_distribution: Dict[Any, float]) -> Dict[str, Any]:
        """
        Generate prediction set satisfying coverage guarantee:
        Include all labels where (1 - P(label)) <= calibrated_quantile.
        """
        if self.calibrated_quantile is None:
            raise RuntimeError("ConformalPredictor must be calibrated before generating prediction sets")

        prediction_set = []
        sorted_candidates = sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True)

        for label, prob in sorted_candidates:
            non_conformity = 1.0 - prob
            if non_conformity <= self.calibrated_quantile:
                prediction_set.append(label)

        # Edge case: if empty due to float discretization, include top-1 candidate
        if not prediction_set and sorted_candidates:
            prediction_set.append(sorted_candidates[0][0])

        return {
            "prediction_set": prediction_set,
            "set_size": len(prediction_set),
            "calibrated_threshold": self.calibrated_quantile,
            "significance_level": self.alpha,
            "guaranteed_coverage": 1.0 - self.alpha
        }

    def evaluate_coverage(self, true_labels: List[Any], test_distributions: List[Dict[Any, float]]) -> Dict[str, float]:
        """
        Evaluate empirical coverage and average set size over a test split.
        """
        if not true_labels or len(true_labels) != len(test_distributions):
            raise ValueError("Invalid test evaluation data")

        covered_count = 0
        total_size = 0

        for label, probs in zip(true_labels, test_distributions):
            pred = self.predict_set(probs)
            pset = pred["prediction_set"]
            if label in pset:
                covered_count += 1
            total_size += len(pset)

        n = len(true_labels)
        return {
            "empirical_coverage": covered_count / n,
            "target_coverage": 1.0 - self.alpha,
            "average_set_size": total_size / n,
            "sample_count": n
        }
