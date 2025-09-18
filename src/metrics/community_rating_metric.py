from .base_metric import BaseMetric
import math


class CommunityRatingMetric(BaseMetric):
    """
    Metric to evaluate community engagement and popularity of a model.

    This metric assesses the community rating based on:
    - Number of likes/stars on the model
    - Download count

    The score is calculated using logarithmic scaling to normalize
    the wide range of possible values.
    """

    def __init__(self, weight: float = 0.15):
        """
        Initialize the CommunityRating metric.

        Args:
            weight (float): Weight of this metric in overall scoring
        """
        super().__init__("CommunityRating", weight)

    def evaluate(self, repo_context: dict) -> float:
        """
        Evaluate community engagement for a model.

        Args:
            repo_context (dict): Dictionary containing model information
                Expected keys:
                - 'hf_likes': Number of likes/stars
                - 'hf_downloads': Number of downloads

        Returns:
            float: Score between 0.0 and 1.0 based on community engagement
        """
        if not repo_context:
            return 0.0

        likes = repo_context.get('hf_likes', 0)
        downloads = repo_context.get('hf_downloads', 0)

        # Handle negative values
        if likes < 0 or downloads < 0:
            raise ValueError("Likes and downloads must be non-negative")

        # Logarithmic scaling for likes
        likes_score = 0.0
        if likes > 0:
            likes_score = math.log10(likes + 1) / 5.0

        # Logarithmic scaling for downloads
        downloads_score = 0.0
        if downloads > 0:
            downloads_score = math.log10(downloads + 1) / 6.0

        # Combined score (simple addition)
        combined_score = likes_score + downloads_score

        return max(0.0, min(1.0, combined_score))

    def get_description(self) -> str:
        """
        Get a description of what this metric measures.

        Returns:
            str: Description of the metric
        """
        return ("Evaluates community engagement through likes/stars and "
                "download counts using logarithmic scaling")
