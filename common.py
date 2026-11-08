"""Shared pieces: Qwen3-Reranker prompt format, batched scoring, nDCG@10.

A Qwen3 reranker reads a fixed prompt with the query and document and answers "yes" or "no".
Its relevance score is P(yes) = sigmoid(logit_yes - logit_no) at the last position. We work
with the logit difference ("score") and convert to a probability only when reporting.
"""

from __future__ import annotations

import math
from typing import Iterable

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


class Encoder:
    """Turns (query, document) pairs into token ids in the official Qwen3-Reranker format."""

    def __init__(self, tok, max_query: int = 128, max_doc: int = 384):
        self.tok = tok
        self.max_query, self.max_doc = max_query, max_doc
        self.prefix = tok(PREFIX, add_special_tokens=False)["input_ids"]
        self.suffix = tok(SUFFIX, add_special_tokens=False)["input_ids"]
        self.yes = tok.convert_tokens_to_ids("yes")
        self.no = tok.convert_tokens_to_ids("no")
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _trunc(self, text: str, limit: int) -> list[int]:
        return self.tok(text, add_special_tokens=False, truncation=True, max_length=limit)["input_ids"]

    def encode(self, query: str, doc: str) -> list[int]:
        head = self.tok(f"<Instruct>: {INSTRUCT}\n<Query>: ", add_special_tokens=False)["input_ids"]
        mid = self.tok("\n<Document>: ", add_special_tokens=False)["input_ids"]
        return self.prefix + head + self._trunc(query, self.max_query) + mid + self._trunc(doc, self.max_doc) + self.suffix


def batch_tensors(torch, seqs: list[list[int]], pad: int, device):
    """Left-padded ids and attention mask, so the last position is the answer slot for every row."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    return ids.to(device), mask.to(device)


def forward_scores(torch, model, enc: Encoder, seqs: list[list[int]], device):
    """yes-minus-no logits at the last position (differentiable)."""
    ids, mask = batch_tensors(torch, seqs, enc.pad, device)
    out = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
    last = out.logits[:, -1, :].float()
    return last[:, enc.yes] - last[:, enc.no]


def score_pairs(torch, model, enc: Encoder, pairs: list[tuple[str, str]], device, tokens_per_batch: int = 32768,
                progress=None) -> list[float]:
    """Scores for many pairs, batched by length to waste little compute on padding."""
    seqs = [enc.encode(q, d) for q, d in pairs]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    scores = [0.0] * len(seqs)
    i = 0
    done = 0
    with torch.inference_mode():
        while i < len(order):
            width = len(seqs[order[i]])
            n = 1
            while i + n < len(order) and (n + 1) * max(width, len(seqs[order[i + n]])) <= tokens_per_batch:
                width = max(width, len(seqs[order[i + n]]))
                n += 1
            idx = order[i:i + n]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                s = forward_scores(torch, model, enc, [seqs[j] for j in idx], device)
            for j, v in zip(idx, s.tolist()):
                scores[j] = v
            i += n
            done += n
            if progress and done // 20000 != (done - n) // 20000:
                progress(done, len(seqs))
    return scores


def ndcg_at_10(ranked: list[str], qrels: dict[str, int]) -> float:
    """nDCG@10 with linear gains, as in trec_eval and BEIR."""
    dcg = sum(qrels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
