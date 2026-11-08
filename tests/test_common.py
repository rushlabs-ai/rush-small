import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import mean, ndcg_at_10  # noqa: E402


class NdcgTests(unittest.TestCase):
    def test_perfect_and_empty(self):
        self.assertAlmostEqual(ndcg_at_10(["a", "b"], {"a": 2, "b": 1}), 1.0)
        self.assertEqual(ndcg_at_10(["x"], {}), 0.0)

    def test_linear_gain_like_trec_eval(self):
        got = ndcg_at_10(["d2", "d1", "d3"], {"d1": 1, "d3": 2})
        want = (1 / math.log2(3) + 2 / math.log2(4)) / (2 + 1 / math.log2(3))
        self.assertAlmostEqual(got, want)

    def test_only_top_10_counts(self):
        ranked = [f"x{i}" for i in range(10)] + ["hit"]
        self.assertEqual(ndcg_at_10(ranked, {"hit": 1}), 0.0)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)
        self.assertEqual(mean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
