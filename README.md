# Rush Small

[![CI](https://github.com/rushlabs-ai/rush-small/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/rushlabs-ai/rush-small/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/rushlabs-ai/rush-small/blob/main/LICENSE)

**A small reranker trained only on data that allows commercial use, reproducible with one command.**
Rush Small is Qwen3-Reranker-0.6B, fine-tuned to match its 4B sibling (distillation), with the whole
recipe, every pinned revision and the evaluation in this repository.

Why bother? The strongest small rerankers today are either non-commercial (jina-reranker v3) or
trained on MS MARCO, whose licence is non-commercial too. Qwen3-Reranker-0.6B is Apache-2.0 and
good. This project asks an honest question: **how much of the 4B teacher's quality can a 0.6B
student recover from commercially usable data alone?** v0 was a draw. v1 is a small, broad gain.

## Results (v1)

nDCG@10 reranking BM25's top 100 on 13 BEIR test sets, up to 1,000 queries per set (a fixed random
sample, the same for every model). From `results/v1-eval.json`:

| Set | Queries | BM25 | Qwen3-Reranker-0.6B | **Rush Small v1** | Change |
|---|---|---|---|---|---|
| TREC-COVID | 50 | 0.593 | 0.832 | **0.836** | +0.4 |
| NFCorpus | 323 | 0.306 | 0.354 | **0.355** | +0.0 |
| FiQA | 648 | 0.233 | 0.367 | **0.400** | +3.3 |
| ArguAna | 1,000 | 0.481 | 0.666 | **0.658** | -0.8 |
| Touché-2020 | 49 | 0.356 | 0.272 | **0.339** | +6.6 |
| Quora | 1,000 | 0.755 | 0.849 | **0.858** | +0.9 |
| DBPedia | 400 | 0.270 | 0.398 | **0.407** | +0.9 |
| SCIDOCS | 1,000 | 0.154 | 0.205 | **0.209** | +0.4 |
| FEVER | 1,000 | 0.484 | 0.813 | **0.820** | +0.7 |
| Climate-FEVER | 1,000 | 0.130 | 0.278 | **0.276** | -0.2 |
| SciFact | 300 | 0.662 | 0.746 | **0.757** | +1.2 |
| **Mean, 11 zero-shot sets** | | 0.402 | 0.525 | **0.538** | **+1.2** |
| NQ (in training data) | 1,000 | 0.273 | 0.539 | **0.572** | +3.3 |
| HotpotQA (in training data) | 1,000 | 0.551 | 0.736 | **0.752** | +1.7 |

Change is in nDCG@10 points. NQ and HotpotQA training questions were part of v1's data, so those
two rows are not zero-shot and are left out of the mean.

**What to read into it.** v1 beats its base model on 9 of the 11 zero-shot sets and gives back
less than a point on ArguAna and Climate-FEVER. The largest gains are on question-style search
(FiQA +3.3) and argument search on Touché (+6.6). Touché and TREC-COVID have only about 50 queries
each, so treat those two rows as noisy. One seed, one run: the mean gain of 1.2 points is
measured, not proven stable across seeds.

Against the teacher on 300 held-out queries from the training mix, the student's top pick matched
the teacher's 81.7% → 85.7% of the time, and listwise KL fell from 0.282 to 0.158.

### v0 for reference

v0 trained on GooAQ only (12,000 questions, lr 1e-5) and was a draw on five BEIR sets: mean 0.466
vs the base model's 0.467 (`results/v0-eval.json`). It gained on FiQA and lost on ArguAna and the
science sets. v1 broadened the data and lowered the learning rate to keep the base model's breadth,
and the losses mostly went away.

## Use

Weights and GGUF: [rushlabs-ai/rush-small-qwen3-reranker-0.6b-v1](https://huggingface.co/rushlabs-ai/rush-small-qwen3-reranker-0.6b-v1)
on Hugging Face, with a Transformers example in the model card (`MODEL_CARD.md` here). The publish
step runs that example against the weights and refuses to upload if it ranks wrongly. The model is a
drop-in replacement for Qwen3-Reranker-0.6B: same prompt format, same yes/no scoring. The GGUF build (q8_0, 639 MB) runs on llama.cpp and
Ollama, and passes [rerankd](https://github.com/rushlabs-ai/rerankd)'s `check` (it keeps the
reranking head and prompt):

```sh
rerankd check rush-small-qwen3-reranker-0.6b-v1-q8_0.gguf
```

## How v1 was made

| Stage | What | Where |
|---|---|---|
| data | 48,000 questions from five sources (below), each with its true passage plus 15 BM25 hard negatives mined from that source's pool, never using any passage known to be relevant. 300 held out | CPU |
| teacher | Qwen3-Reranker-4B scores all 763,200 pairs (yes-minus-no logit), in resumable shards | 1 × L40S, 3.3 h |
| train | full fine-tune of Qwen3-Reranker-0.6B: listwise KL to the teacher's scores + 0.5 × pointwise BCE, lr 5e-6, 2,300 steps of 8 questions (18,400 questions seen), checkpoint every 500 steps | 1 × L40S, 4.0 h |
| eval | BEIR sets trimmed to the sampled queries and their BM25 top 100 on CPU, then nDCG@10 per model, each result saved as it finishes | CPU, then 1 × L40S per model |
| export | GGUF q8_0 with llama.cpp b11980 (pinned), then `rerankd check` | CPU |

```sh
modal run pipeline.py --version v1 --stage all --smoke    # a few cents, end to end
modal run pipeline.py --version v1 --stage all --steps 2300
modal run pipeline.py --stage all                         # v0, reproduces the earlier run
RUSH_SMALL_HF_SECRET=<modal secret with HF_TOKEN> modal run pipeline.py --version v1 --stage publish
```

Long stages resume: the teacher writes shards, training saves a checkpoint every 500 steps and
picks it up on a rerun, and evaluation skips any (model, set) result already saved. Every model
and dataset is pinned to a commit in `pipeline.py`. The Python environment is hash-locked
(`requirements.txt`, nothing newer than 7 days at lock time). llama.cpp is pinned by commit, and
rerankd by release checksum.

## Data and licences

| Item | Licence | Used for |
|---|---|---|
| [GooAQ](https://github.com/allenai/gooaq) via sentence-transformers/gooaq, 16,000 questions | Apache-2.0 | training |
| [PubMedQA](https://github.com/pubmedqa/pubmedqa) (pqa_artificial), 8,000 questions | MIT | training |
| [Natural Questions](https://ai.google.com/research/NaturalQuestions) via sentence-transformers/natural-questions, 10,000 questions | CC BY-SA 3.0 | training |
| [HotpotQA](https://hotpotqa.github.io/) via sentence-transformers/hotpotqa, 8,000 questions | CC BY-SA 4.0 | training |
| [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) via sentence-transformers/squad, 6,000 questions | CC BY-SA 4.0 | training |
| Qwen3-Reranker-0.6B (student), Qwen3-Reranker-4B (teacher) | Apache-2.0 | base model, soft labels |
| BEIR test sets | per dataset, see [BEIR](https://github.com/beir-cellar/beir) | evaluation only |

Every training source allows commercial use. Three are share-alike (CC BY-SA). Whether share-alike
terms reach model weights trained on the data is not settled, so check with your own counsel if
that matters for you. v0 used GooAQ only (Apache-2.0). No MS MARCO, no data generated by a closed
model.

## Limits

- One seed and one learning rate per version. v1 trained on 39% of one pass over its data, for
  budget reasons. A full pass might do better or worse.
- English-heavy training data. The base model's multilingual ability is not measured here.
- BEIR numbers depend on the BM25 first stage (bm25s, English stopwords) and on the query sample.
  They are comparable within these tables, not directly with other papers' tables.
- The teacher put the true passage first for 81% of training questions (71% on GooAQ and NQ), so
  some of what the student learned is the teacher's mistakes. `results/v1-teacher.json` has the
  breakdown by source.

Built by [Rush Labs](https://rushlabs.ai), an applied AI lab. Apache-2.0.
