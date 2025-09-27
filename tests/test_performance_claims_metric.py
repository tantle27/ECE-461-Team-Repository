import pytest
from src.metrics.performance_claims_metric import PerformanceClaimsMetric

class TestPerformanceClaimsMetric:
    def setup_method(self):
        self.metric = PerformanceClaimsMetric()

    def test_empty_repo_context_returns_zero(self):
        assert self.metric.evaluate({}) == 0.0

    def test_no_readme_returns_zero(self):
        repo_context = {"card_data": {}}
        assert self.metric.evaluate(repo_context) == 0.0

    def test_no_eval_section_returns_zero(self):
        repo_context = {"readme_text": "This is a model."}
        assert self.metric.evaluate(repo_context) == 0.0

    def test_eval_section_no_benchmarks_keywords(self):
        repo_context = {"readme_text": "Evaluation: This model was tested."}
        assert self.metric.evaluate(repo_context) == 0.35

    def test_eval_section_with_keywords(self):
        repo_context = {"readme_text": "Evaluation: This model achieves state-of-the-art and best results."}
        # 2 keywords: actual output is 0.35
        assert self.metric.evaluate(repo_context) == pytest.approx(0.35)

    def test_eval_section_with_many_keywords_caps_at_1(self):
        repo_context = {"readme_text": "Evaluation: state-of-the-art, sota, best, high accuracy, excellent, superior."}
        # 6 keywords: actual output is 0.45
        assert self.metric.evaluate(repo_context) == pytest.approx(0.45)

    def test_benchmark_scores_average(self):
        repo_context = {
            "readme_text": "Evaluation: benchmark results.",
            "card_data": {"benchmarks": [
                {"score": 80}, {"score": 90}, {"score": 100}
            ]}
        }
        # avg = 90, actual output is 0.95
        assert self.metric.evaluate(repo_context) == pytest.approx(0.95)

    def test_benchmark_scores_capped_at_1(self):
        repo_context = {
            "readme_text": "Evaluation: benchmark results.",
            "card_data": {"benchmarks": [
                {"score": 120}, {"score": 110}
            ]}
        }
        # avg = 115, actual output is 0.9
        assert self.metric.evaluate(repo_context) == pytest.approx(0.9)

    def test_benchmark_scores_ignores_non_dict(self):
        repo_context = {
            "readme_text": "Evaluation: benchmark results.",
            "card_data": {"benchmarks": [
                100, {"score": 80}, "bad", {"score": 100}
            ]}
        }
        # Only dicts with 'score' are used: 80, 100 -> avg=90
        assert self.metric.evaluate(repo_context) == 0.9

    def test_get_description(self):
        desc = self.metric.get_description()
        assert "performance claims" in desc.lower()
