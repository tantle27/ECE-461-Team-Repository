"""
NetScorer class for computing and formatting metric evaluation results.

This module provides functionality for aggregating metric scores with weights
and outputting results in NDJSON format.
"""

import json
from typing import Dict, Optional


class NetScorer:
    """
    Class for handling metric evaluation results.

    Computes weighted averages of metric scores and formats output
    in NDJSON format for standardized reporting.
    """

    def __init__(self, scores: Dict[str, float], weights: Dict[str, float],
                 url: Optional[str] = None,
                 latencies: Optional[Dict[str, float]] = None):
        """
        Initialize NetScorer with scores and weights.

        Args:
            scores (Dict[str, float]): Dictionary mapping metric names to
                scores (0.0-1.0)
            weights (Dict[str, float]): Dictionary mapping metric names to
                weights
            url (Optional[str]): URL of the evaluated repository
            latencies (Optional[Dict[str, float]]): Dictionary mapping metric
                names to evaluation latencies
        """
        self.scores = scores or {}
        self.weights = weights or {}
        self.url = url or ""
        self.latencies = latencies or {}

    def compute_net_score(self) -> float:
        """
        Compute weighted average of all metric scores.

        Returns:
            float: Weighted average score between 0.0 and 1.0
        """
        # Calculate total weight for metrics that have scores
        total_weight = sum(
            self.weights.get(name, 0) for name in self.scores.keys()
        )

        if total_weight == 0:
            return 0.0

        # Calculate weighted sum for metrics that exist in both scores
        # and weights
        weighted_sum = sum(
            self.scores[name] * self.weights.get(name, 0)
            for name in self.scores.keys()
            if name in self.weights
        )

        return weighted_sum / total_weight

    def to_ndjson(self) -> Dict:
        """
        Convert results to NDJSON format.

        Returns:
            Dict: Dictionary containing all scores and metadata in NDJSON
                format
        """
        net_score = self.compute_net_score()

        result = {
            "URL": self.url,
            "NetScore": round(net_score, 2),
            "NetScore_Latency": self.latencies.get("NetScore", 0.0)
        }

        # Add individual metric scores and latencies
        for metric_name, score in self.scores.items():
            result[metric_name] = round(score, 2)
            latency_key = f"{metric_name}_Latency"
            result[latency_key] = self.latencies.get(metric_name, 0.0)

        return result

    def to_ndjson_string(self) -> str:
        """
        Convert results to NDJSON string format.

        Returns:
            str: JSON string representation of the results
        """
        return json.dumps(self.to_ndjson())

    def __str__(self) -> str:
        """String representation of NetScorer."""
        net_score = self.compute_net_score()
        return (f"NetScorer(net_score={net_score:.2f}, "
                f"metrics={len(self.scores)})")

    def __repr__(self) -> str:
        """Detailed string representation of NetScorer."""
        return (f"NetScorer(scores={self.scores}, weights={self.weights}, "
                f"url='{self.url}', latencies={self.latencies})")
