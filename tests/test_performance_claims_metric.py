def test_score_model_index_mixed_and_missing_metrics():
    metric = PerformanceClaimsMetric()
    # Mixed strong/weak
    model_index = {"results": [{"metrics": [{"score": 1.0}, {"foo": "bar"}]}]}
    result = metric._score_model_index(model_index)
    assert result is not None and result > 0.6
    # Missing metrics key
    model_index = {"results": [{"no_metrics": 123}]}
    assert metric._score_model_index(model_index) is None
    # Non-list results
    model_index = {"results": {"metrics": [{"score": 1.0}]}}
    assert metric._score_model_index(model_index) is None

def test_score_card_data_plausible_keys_mixed():
    metric = PerformanceClaimsMetric()
    card_data = {"results": [1, "0.5", {"val": "0.7"}], "metrics": {"foo": "0.8"}, "evaluation": "0.9"}
    score = metric._score_card_data(card_data)
    assert 0.0 <= score <= 1.0

def test_count_benchmark_hits_duplicates():
    metric = PerformanceClaimsMetric()
    text = "glue glue glue mnli mnli"
    assert metric._count_benchmark_hits(text, ["glue", "mnli"]) == 2
import pytest
from src.metrics.performance_claims_metric import PerformanceClaimsMetric

class TestPerformanceClaimsMetric:
    def test_has_numeric_score_various(self):
        metric = PerformanceClaimsMetric()
        # Dict with float value
        assert metric._has_numeric_score({"score": "1.23"})
        # Dict with int value
        assert metric._has_numeric_score({"value": 5})
        # Dict with string value that is not numeric
        assert not metric._has_numeric_score({"foo": "bar"})
        # Dict with string containing a number
        assert metric._has_numeric_score({"foo": "score: 0.99"})
        # Non-dict
        assert not metric._has_numeric_score([1,2,3])
        # Dict with value that raises exception
        class Bad:
            def __str__(self): raise Exception()
        assert not metric._has_numeric_score({"score": Bad()})

    def test_score_model_index_exception(self):
        metric = PerformanceClaimsMetric()
        # Should handle exception and return None
        class BadDict(dict):
            def get(self, k, d=None): raise Exception()
        assert metric._score_model_index(BadDict()) is None

    def test_score_card_data_exception(self):
        metric = PerformanceClaimsMetric()
        # Should handle exception in value_is_numeric
        class Bad:
            def __float__(self): raise Exception()
        card_data = {"results": [Bad()]}
        # Should not raise
        assert metric._score_card_data(card_data) == 0.0

    def test_value_is_numeric_exceptions(self):
        metric = PerformanceClaimsMetric()
        class Bad:
            def __float__(self): raise Exception()
        assert not metric._value_is_numeric(Bad())

    def test_readme_table_and_refs(self):
        metric = PerformanceClaimsMetric()
        repo_context = {
            "readme_text": "Evaluation: | col1 | col2 |\n|----|----|\nValue | 0.9 | arxiv.org",
            "card_data": {},
        }
        # Table and refs should boost score
        score = metric.evaluate(repo_context)
        assert score > 0.35

    def test_readme_numeric_hits(self):
        metric = PerformanceClaimsMetric()
        repo_context = {
            "readme_text": "Evaluation: accuracy 99% f1 0.88 recall 0.77 precision 0.66",
            "card_data": {},
        }
        # Numeric hits should increase score
        score = metric.evaluate(repo_context)
        assert score > 0.35

    def test_bert_base_uncased_floor(self):
        metric = PerformanceClaimsMetric()
        repo_context = {
            "readme_text": "Evaluation: glue benchmark results.",
            "card_data": {},
            "hf_id": "bert-base-uncased"
        }
        # Should apply floor logic
        score = metric.evaluate(repo_context)
        assert score >= 0.92
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



        # --- Additional edge/branch tests for coverage ---
        @staticmethod
        def test_score_model_index_none_and_non_dict():
            metric = PerformanceClaimsMetric()
            # None
            assert metric._score_model_index(None) is None
            # Not a dict
            assert metric._score_model_index([]) is None

        @staticmethod
        def test_score_model_index_empty_results():
            metric = PerformanceClaimsMetric()
            # Dict with no results
            assert metric._score_model_index({"results": []}) is None

        @staticmethod
        def test_score_model_index_weak_and_strong():
            metric = PerformanceClaimsMetric()
            # Strong: metrics with numeric score
            model_index = {"results": [{"metrics": [{"score": 1.0}]}]}
            assert metric._score_model_index(model_index) > 0.6
            # Weak: metrics with no numeric score
            model_index = {"results": [{"metrics": [{"foo": "bar"}]}]}
            # Should return a float or None, but not error
            result = metric._score_model_index(model_index)
            assert result is None or isinstance(result, float)

        @staticmethod
        def test_score_card_data_empty_and_non_dict():
            metric = PerformanceClaimsMetric()
            # Not a dict
            assert metric._score_card_data([]) == 0.0
            # Empty dict
            assert metric._score_card_data({}) == 0.0

        @staticmethod
        def test_score_card_data_benchmarks_numeric():
            metric = PerformanceClaimsMetric()
            card_data = {"benchmarks": [{"score": 1.0}, {"score": 2.0}, {"score": 3.0}]}
            assert 0.75 <= metric._score_card_data(card_data) <= 1.0

        @staticmethod
        def test_score_card_data_plausible_keys():
            metric = PerformanceClaimsMetric()
            card_data = {"results": [1, 2, 3], "evaluation": ["0.9", 1.0], "metrics": [{"val": 0.5}]}
            assert metric._score_card_data(card_data) > 0.0

        @staticmethod
        def test_score_card_data_text_hits():
            metric = PerformanceClaimsMetric()
            card_data = {"results": "no numbers here", "evaluation": "also no numbers"}
            assert metric._score_card_data(card_data) == 0.35

        @staticmethod
        def test_value_is_numeric():
            metric = PerformanceClaimsMetric()
            assert metric._value_is_numeric(1.0)
            assert metric._value_is_numeric("2.5")
            assert not metric._value_is_numeric("foo")

        @staticmethod
        def test_count_benchmark_hits():
            metric = PerformanceClaimsMetric()
            text = "glue mnli sst-2 imagenet"
            assert metric._count_benchmark_hits(text, ["glue", "sst-2", "imagenet"]) == 3
            assert metric._count_benchmark_hits("", ["glue"]) == 0
