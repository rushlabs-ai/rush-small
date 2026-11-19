import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import Encoder, mean, ndcg_at_10  # noqa: E402


class FakeTokenizer:
    """Character-level stand-in for a Hugging Face tokenizer (str or list input, truncation)."""

    pad_token_id = 0
    eos_token_id = 1

    def _one(self, text, truncation=False, max_length=None):
        ids = [ord(c) for c in text]
        return ids[:max_length] if truncation and max_length else ids

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        if isinstance(text, list):
            return {"input_ids": [self._one(t, truncation, max_length) for t in text]}
        return {"input_ids": self._one(text, truncation, max_length)}

    def convert_tokens_to_ids(self, token):
        return {"yes": 2, "no": 3}[token]


class EncoderTests(unittest.TestCase):
    def test_encode_many_matches_encode(self):
        enc = Encoder(FakeTokenizer(), max_query=5, max_doc=7)
        pairs = [("what is rust", "a systems language"), ("q", "d"), ("what is rust", "another long document"),
                 ("", "empty query"), ("q", "d")]
        self.assertEqual(enc.encode_many(pairs, chunk=2), [enc.encode(q, d) for q, d in pairs])


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
