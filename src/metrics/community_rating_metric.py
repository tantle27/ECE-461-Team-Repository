from .base_metric import BaseMetric


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
        Evaluate community engagement using specific thresholds.

        Args:
            repo_context (dict): Dictionary containing model information
                Expected keys:
                - 'hf_likes': Number of likes/stars
                - 'hf_downloads': Number of downloads

        Returns:
            float: Score between 0.0 and 1.0 based on likes and downloads:
                  Likes: 0:<5:0.1, <10:0.2, <50:0.3, <100:0.4, >100:0.5
                  Downloads: <1k:0, <5k:0.1, <10k:0.2, <50k:0.3,
                            <100k:0.4, >100k:0.5
        """
        if not repo_context:
            return 0.0

        likes = repo_context.get('hf_likes', 0)
        downloads = repo_context.get('hf_downloads', 0)

        # Handle negative values
        if likes < 0 or downloads < 0:
            raise ValueError("Likes and downloads must be non-negative")

        # Likes scoring with specific thresholds
        if likes == 0:
            likes_score = 0.0
        elif likes < 5:
            likes_score = 0.1
        elif likes < 10:
            likes_score = 0.2
        elif likes < 50:
            likes_score = 0.3
        elif likes < 100:
            likes_score = 0.4
        else:
            likes_score = 0.5

        # Downloads scoring with specific thresholds (in thousands)
        downloads_k = downloads / 1000.0
        if downloads_k < 1:
            downloads_score = 0.0
        elif downloads_k < 5:
            downloads_score = 0.1
        elif downloads_k < 10:
            downloads_score = 0.2
        elif downloads_k < 50:
            downloads_score = 0.3
        elif downloads_k < 100:
            downloads_score = 0.4
        else:
            downloads_score = 0.5

        # Combined score (sum of both components)
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
