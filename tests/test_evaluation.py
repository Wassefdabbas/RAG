import pytest
"""
Tests for evaluation metrics (Recall@K, MRR).
Pure logic — no database or LLM calls needed.
"""

from src.evaluation.metrics import recall_at_k, reciprocal_rank


class TestRecallAtK:
    def test_relevant_doc_in_top_k_returns_1(self):
        retrieved = [10, 3, 8, 5, 1]
        relevant = {8}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_relevant_doc_outside_top_k_returns_0(self):
        retrieved = [10, 3, 8, 5, 1, 7]
        relevant = {7}
        assert recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_no_relevant_docs_at_all_returns_0(self):
        retrieved = [10, 3, 8]
        relevant = {99}
        assert recall_at_k(retrieved, relevant, k=5) == 0.0

    def test_multiple_relevant_docs_any_match_counts(self):
        retrieved = [10, 3, 8]
        relevant = {3, 4}
        assert recall_at_k(retrieved, relevant, k=5) == 1.0

    def test_k_smaller_than_relevant_position_returns_0(self):
        retrieved = [10, 3, 8, 5, 1]
        relevant = {1}  # at position 5
        assert recall_at_k(retrieved, relevant, k=3) == 0.0


class TestReciprocalRank:
    def test_relevant_doc_first_gives_rr_1(self):
        retrieved = [4, 10, 8]
        relevant = {4}
        assert reciprocal_rank(retrieved, relevant) == 1.0

    def test_relevant_doc_third_gives_rr_one_third(self):
        retrieved = [10, 8, 4]
        relevant = {4}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_no_relevant_doc_gives_rr_0(self):
        retrieved = [10, 8, 4]
        relevant = {99}
        assert reciprocal_rank(retrieved, relevant) == 0.0

    def test_uses_first_match_only(self):
        retrieved = [10, 4, 4, 8]
        relevant = {4}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 2)