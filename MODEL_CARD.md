---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: text-ranking
library_name: transformers
tags:
- reranker
- cross-encoder
- distillation
- qwen3
- gguf
datasets:
- sentence-transformers/gooaq
- qiaojin/PubMedQA
- sentence-transformers/natural-questions
- sentence-transformers/hotpotqa
- sentence-transformers/squad
---

# Rush Small v1 (Qwen3-Reranker-0.6B, distilled)

A 0.6B reranker trained only on data that allows commercial use. It is
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) fine-tuned to match
[Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) on 48,000 questions from five
sources. The whole recipe, every pinned revision and every result file are in
[rushlabs-ai/rush-small](https://github.com/rushlabs-ai/rush-small), and one command reproduces it.

It is a drop-in replacement for Qwen3-Reranker-0.6B: same prompt, same yes/no scoring, same size.
This repository also has a GGUF build (q8_0) for llama.cpp.

## Results

nDCG@10 reranking BM25's top 100 on BEIR test sets, up to 1,000 queries per set (a fixed random
sample, the same for every model):

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

Up on 9 of the 11 zero-shot sets, slightly behind on ArguAna and Climate-FEVER. Touché and
TREC-COVID have about 50 queries each, so treat those rows as noisy. One seed, one run.

## Use with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "rushlabs-ai/rush-small-qwen3-reranker-0.6b-v1"
tok = AutoTokenizer.from_pretrained(name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()
yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
          "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
instruct = "Given a web search query, retrieve relevant passages that answer the query"

def rerank(query, docs):
    texts = [f"{prefix}<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {d}{suffix}" for d in docs]
    batch = tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    with torch.no_grad():
        last = model(**batch).logits[:, -1, :]
    return torch.sigmoid(last[:, yes] - last[:, no]).tolist()  # relevance, 0 to 1

docs = ["Tides are caused mainly by the Moon's gravity pulling on the Earth's oceans.",
        "Tide is a brand of laundry detergent sold in the United States."]
print(rerank("what causes tides", docs))  # the first passage scores far higher
```

## Use with llama.cpp, Ollama and rerankd

`rush-small-qwen3-reranker-0.6b-v1-q8_0.gguf` (639 MB) keeps the reranking head and prompt, and
passes [rerankd](https://github.com/rushlabs-ai/rerankd)'s check:

```sh
rerankd check rush-small-qwen3-reranker-0.6b-v1-q8_0.gguf
llama-server -m rush-small-qwen3-reranker-0.6b-v1-q8_0.gguf --rerank
```

rerankd adds rerank endpoints (`/api/rerank`, `/v1/rerank`, `/rerank`) to Ollama.

## Training

- **Data:** 48,000 questions, each with its true passage and 15 BM25 hard negatives mined from
  the same source, never using a passage known to be relevant. 300 held out.
- **Teacher:** Qwen3-Reranker-4B scores all 763,200 pairs (yes-minus-no logit).
- **Student:** full fine-tune of Qwen3-Reranker-0.6B with listwise KL to the teacher plus
  0.5 × pointwise BCE, lr 5e-6, 2,300 steps of 8 questions, on one NVIDIA L40S (4 hours).
- **Held out:** the student's top pick matched the teacher's 81.7% → 85.7% of the time.

## Data and licences

| Source | Questions | Licence |
|---|---|---|
| GooAQ | 16,000 | Apache-2.0 |
| PubMedQA (pqa_artificial) | 8,000 | MIT |
| Natural Questions | 10,000 | CC BY-SA 3.0 |
| HotpotQA | 8,000 | CC BY-SA 4.0 |
| SQuAD | 6,000 | CC BY-SA 4.0 |

Every source allows commercial use. Three are share-alike. Whether share-alike terms reach model
weights trained on the data is not settled, so check with your own counsel if that matters for you.
No MS MARCO, no data generated by a closed model. The model is released under Apache-2.0, like its
base model.

## Limits

- One seed and one learning rate. Trained on 39% of one pass over its data.
- English-heavy training data. Multilingual quality is not measured.
- BEIR numbers depend on the BM25 first stage and the query sample. They compare within this table,
  not directly with other papers.

Built by [Rush Labs](https://rushlabs.ai), an applied AI lab.
