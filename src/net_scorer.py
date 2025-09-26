"""
NetScorer class for computing and formatting metric evaluation results.

This module provides functionality for aggregating metric scores with weights
and outputting results in NDJSON format.
"""

import json
from typing import Dict, Optional
from repo_context import RepoContext
import logging

logger = logging.getLogger("acme-cli.net_scorer")


class NetScorer:
    """
    Class for handling metric evaluation results.

    Computes weighted averages of metric scores and formats output
    in NDJSON format for standardized reporting.
    """

    def __init__(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float],
        url: Optional[str] = None,
        latencies: Optional[Dict[str, float]] = None,
    ):
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
            "NetScore_Latency": self.latencies.get("NetScore", 0.0),
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
        return (
            f"NetScorer(net_score={net_score:.2f}, "
            f"metrics={len(self.scores)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation of NetScorer."""
        return (
            f"NetScorer(scores={self.scores}, weights={self.weights}, "
            f"url='{self.url}', latencies={self.latencies})"
        )


def _emit_ndjson(
    ctx: RepoContext,
    category: str,
    per_metric: Dict[str, float],
    net: float,
    per_metric_lat_ms: Dict[str, int],
    net_latency_ms: int,
) -> None:
    """
    Build the NDJSON payload, log context.
    """
    if ctx.hf_id:
        name = ctx.hf_id.split("/")[-1]
    elif ctx.url:
        name = ctx.url.rstrip("/").split("/")[-1] or ctx.url
    else:
        name = "unknown"

    # Convenience getters
    def get(k, d=0.0):
        return float(per_metric.get(k, d))

    def lat(k):
        return int(per_metric_lat_ms.get(k, 0))

    size = get("Size", 0.0)
    size_score = {
        "raspberry_pi": max(0.0, min(1.0, size - 0.10)),
        "jetson_nano": max(0.0, min(1.0, size - 0.05)),
        "desktop_pc": max(0.0, min(1.0, size)),
        "aws_server": max(0.0, min(1.0, size)),
    }

    da = get("DatasetAvailability", 0.0)
    cq = get("CodeQuality", 0.0)
    ds_code = round((da + cq) / 2.0, 2)

    nd = {
        "name": name,
        "category": category,
        "net_score": round(float(net), 2),
        "net_score_latency": int(net_latency_ms),
        "ramp_up_time": round(get("RampUpTime"), 2),
        "ramp_up_time_latency": lat("RampUpTime"),
        "bus_factor": round(get("BusFactor"), 2),
        "bus_factor_latency": lat("BusFactor"),
        "performance_claims": round(get("PerformanceClaims"), 2),
        "performance_claims_latency": lat("PerformanceClaims"),
        "license": round(get("License"), 2),
        "license_latency": lat("License"),
        "size_score": size_score,
        "size_score_latency": lat("Size"),
        "dataset_and_code_score": ds_code,
        "dataset_and_code_score_latency": max(
            lat("DatasetAvailability"), lat("CodeQuality")
        ),
        "dataset_quality": round(get("DatasetQuality"), 2),
        "dataset_quality_latency": lat("DatasetQuality"),
        "code_quality": round(get("CodeQuality"), 2),
        "code_quality_latency": lat("CodeQuality"),
    }

    # ---- Logging (file) ----
    # Debug detail: raw latencies + full NDJSON payload
    logger.debug("metric_latencies_ms=%s", per_metric_lat_ms)
    logger.info(
        "NDJSON summary: name=%s category=%s net=%.2f (net_latency_ms=%d)",
        name,
        category,
        nd["net_score"],
        nd["net_score_latency"],
    )
    logger.debug("NDJSON payload=%s", nd)

    # ---- Emission (stdout) ----
    print(json.dumps(nd, separators=(",", ":")))
