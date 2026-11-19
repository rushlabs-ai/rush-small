"""Rush Small: distil Qwen3-Reranker-4B into Qwen3-Reranker-0.6B with permissively licensed data.

    modal run pipeline.py --stage all --smoke     # tiny end-to-end check, a few cents
    modal run pipeline.py --stage all             # the real run
    modal run pipeline.py --stage eval            # re-run one stage (data|teacher|train|eval|export)
    modal run pipeline.py --version v1 --stage all --smoke   # v1: broader permissive data mix, 13 BEIR sets

Stages, all cached in the Modal Volume "rush-small":
  data     GooAQ (Apache-2.0) questions. Candidates are the true answer + BM25 hard negatives
  teacher  Qwen3-Reranker-4B scores every candidate (the soft labels)
  train    full fine-tune of Qwen3-Reranker-0.6B to match the teacher (listwise KL + pointwise BCE)
  eval     nDCG@10 on five BEIR test sets, reranking BM25's top 100: BM25, base 0.6B, ours
  export   GGUF (llama.cpp b11980, pinned) and a `rerankd check` that it is served as a reranker
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
REV = {  # every model and dataset pinned to a commit
    "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
    "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    "sentence-transformers/gooaq": "b089f728748a068b7bc5234e5bcf5b25e3c8279c",
    "BeIR/scifact": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
    "BeIR/scifact-qrels": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
    "BeIR/nfcorpus": "b5026a0e96e8a7ac4f95f482a596389289d46269",
    "BeIR/nfcorpus-qrels": "a451b3b26d3ae1358f259c1a3a4dd61fcea35a65",
    "BeIR/fiqa": "979c07a7cb5ccc6ca009792241fa1250b98055dd",
    "BeIR/fiqa-qrels": "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
    "BeIR/arguana": "9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f",
    "BeIR/arguana-qrels": "ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7",
    "BeIR/scidocs": "acebceec772628c6679d76d6cff30ad517d17f4e",
    "BeIR/scidocs-qrels": "735ea1048e37b1ebce14c6dc3d33a5edaf66d3dc",
    # v1 training data (licence of the original dataset in the comment)
    "sentence-transformers/natural-questions": "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17",  # CC BY-SA 3.0
    "sentence-transformers/hotpotqa": "f07d3cd2d290ea2e83ed35e33d67d6a4658b8786",  # CC BY-SA 4.0
    "sentence-transformers/squad": "d84c8c2ef64693264c890bb242d2e73fc0a46c40",  # CC BY-SA 4.0
    "qiaojin/PubMedQA": "9001f2853fb87cab8d220904e0de81ac6973b318",  # MIT
    # v1 evaluation: the rest of the public BEIR test sets
    "BeIR/trec-covid": "7e16fde3016c639c7f856e803f4bab92645562c4",
    "BeIR/trec-covid-qrels": "532ac68ee6756ac22c9346eebf65bd3c6a042e10",
    "BeIR/nq": "b7253e6c379163d024ddb1d6948152a91a2e3b46",
    "BeIR/nq-qrels": "519acd4e48bb3e5da22b2b888ce36c614f4f2bc9",
    "BeIR/hotpotqa": "a7e8bab212f5a89f9be1bc9b654aa6dfa317f32b",
    "BeIR/hotpotqa-qrels": "b15429e9244c8ec966985d7778427c3b1543b314",
    "BeIR/webis-touche2020": "7ebed360c2743a0024cb96a404eb89e27c931ebc",
    "BeIR/webis-touche2020-qrels": "dea3c4c7339f61c3e1abc42b9bdbf337b115aa97",
    "BeIR/quora": "54934c6ffc800b0928b057f047315b5d1a1d71d8",
    "BeIR/quora-qrels": "7906857d63e4b4d41fd16c954c929c3ff3c580d9",
    "BeIR/dbpedia-entity": "6029bd92b4a912ce92b2b0972999e297f76dbcbc",
    "BeIR/dbpedia-entity-qrels": "c81f61ac54bbd5f162f9dc6ad36e236e2aeb8d82",
    "BeIR/fever": "426aea4d30c65c041f093ae072e25fb5a0e64659",
    "BeIR/fever-qrels": "64cd0e4ef9b63a88ac8e69d1133a2f649acd4745",
    "BeIR/climate-fever": "d4edc15f76aef202e29b730b4d6801a7dd70df01",
    "BeIR/climate-fever-qrels": "ca2c7ed51cf8b40c10c23a099a27eceb3678156b",
}
BEIR = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]
# v1 reports all 13 public BEIR test sets. NQ and HotpotQA share a source with v1 training data, so they are
# reported separately and left out of the zero-shot mean.
BEIR_V1 = ["trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020", "quora",
           "dbpedia-entity", "scidocs", "fever", "climate-fever", "scifact"]
NOT_ZERO_SHOT = {"nq", "hotpotqa"}
NAMES = {"v0": "rush-small-qwen3-reranker-0.6b-v0", "v1": "rush-small-qwen3-reranker-0.6b-v1"}
NAME = NAMES["v0"]  # "qwen3-reranker" lets llama.cpp detect the format
# v1 training mix: queries per source (positives and BM25 negatives come from the same source)
SOURCES_V1 = {"gooaq": 16000, "natural-questions": 10000, "hotpotqa": 8000, "squad": 6000, "pubmedqa": 8000}
LLAMA_CPP = ("b11980", "f7c01c5226cac1de1052c17920ff38cdefea0ae6")
RERANKD = ("v0.1.0", "rerankd_0.1.0_linux_amd64.tar.gz", 723045009)  # tag, asset, GitHub asset id
RERANKD_SHA256_PINNED = "716b8994110a5f46d67d97aa53fd5dc023f20a34c1face59f973c4a504b04083"  # from the release's SHA256SUMS

volume = modal.Volume.from_name("rush-small", create_if_missing=True)
app = modal.App("rush-small")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("common")
)
RERANKD_LOCAL = HERE / ".cache" / RERANKD[1]


def _fetch_rerankd() -> None:
    """Download the rerankd release on this machine (Modal's shared IPs get rate-limited by GitHub)."""
    import hashlib
    import urllib.request

    if RERANKD_LOCAL.exists():
        return
    RERANKD_LOCAL.parent.mkdir(exist_ok=True)
    req = urllib.request.Request(f"https://api.github.com/repos/rushlabs-ai/rerankd/releases/assets/{RERANKD[2]}",
                                 headers={"Accept": "application/octet-stream"})
    data = urllib.request.urlopen(req, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != RERANKD_SHA256_PINNED:
        raise SystemExit("rerankd download does not match its pinned SHA-256")
    RERANKD_LOCAL.write_bytes(data)


if modal.is_local():
    _fetch_rerankd()

export_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ca-certificates")
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_CPP[0]} https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        f'test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "{LLAMA_CPP[1]}"',
    )
    .pip_install_from_requirements(str(HERE / "requirements-gguf.txt"))
    .add_local_file(RERANKD_LOCAL, f"/opt/{RERANKD[1]}")
)

V = Path("/vol")


def _dl(repo: str, filename: str, repo_type: str = "model") -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type=repo_type, revision=REV.get(repo))


def _model(repo: str, device):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = repo if Path(repo).is_dir() else snapshot_download(repo, revision=REV.get(repo),
                                                              allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    return tok, model


# -- data ----------------------------------------------------------------------------


@app.function(image=image, cpu=8, memory=32768, volumes={"/vol": volume}, timeout=2 * 3600)
def build_data(n_queries: int, n_cand: int, corpus_size: int, tag: str) -> dict:
    import bm25s
    import pyarrow.parquet as pq

    rng = random.Random(7)
    pairs = []
    for f in ("pair/train-00000-of-00002.parquet", "pair/train-00001-of-00002.parquet"):
        t = pq.read_table(_dl("sentence-transformers/gooaq", f, "dataset"), columns=["question", "answer"])
        pairs += list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))
    pairs = [(q.strip(), a.strip()) for q, a in pairs if q and a and len(a) > 20]
    rng.shuffle(pairs)
    pool = pairs[:corpus_size]
    corpus = list(dict.fromkeys(a for _, a in pool))
    index_of = {a: i for i, a in enumerate(corpus)}
    queries = pool[: n_queries + 300]  # 300 held out to validate against the teacher
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus, stopwords="en", show_progress=False), show_progress=False)
    hits, _ = retriever.retrieve(bm25s.tokenize([q for q, _ in queries], stopwords="en", show_progress=False),
                                 k=40, show_progress=False, n_threads=8)
    rows = []
    for (q, a), ranked in zip(queries, hits.tolist()):
        pos = index_of[a]
        negatives = [corpus[i] for i in ranked if i != pos][: n_cand - 1]
        if len(negatives) < n_cand - 1:
            continue
        docs = [a] + negatives
        rows.append({"query": q, "docs": docs})
    out = V / "data" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[300:]))
    (out / "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:300]))
    volume.commit()
    return {"train_queries": len(rows) - 300, "heldout": 300, "corpus": len(corpus), "candidates": n_cand}


def _source_pairs(source: str) -> list[tuple[str, str]]:
    """(query, relevant passage) pairs from one permissively licensed source. A query may appear more than once."""
    import pyarrow.parquet as pq

    def table(repo, f, cols):
        return pq.read_table(_dl(repo, f, "dataset"), columns=cols)

    if source == "gooaq":  # Apache-2.0
        out = []
        for f in ("pair/train-00000-of-00002.parquet", "pair/train-00001-of-00002.parquet"):
            t = table("sentence-transformers/gooaq", f, ["question", "answer"])
            out += zip(t.column("question").to_pylist(), t.column("answer").to_pylist())
        return out
    if source == "natural-questions":  # CC BY-SA 3.0
        t = table("sentence-transformers/natural-questions", "pair/train-00000-of-00001.parquet", ["query", "answer"])
        return list(zip(t.column("query").to_pylist(), t.column("answer").to_pylist()))
    if source == "hotpotqa":  # CC BY-SA 4.0. Each question has two supporting paragraphs
        t = table("sentence-transformers/hotpotqa", "triplet/train-00000-of-00001.parquet", ["anchor", "positive"])
        return list(zip(t.column("anchor").to_pylist(), t.column("positive").to_pylist()))
    if source == "squad":  # CC BY-SA 4.0. The "answer" column is the context paragraph
        t = table("sentence-transformers/squad", "pair/train-00000-of-00001.parquet", ["question", "answer"])
        return list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))
    if source == "pubmedqa":  # MIT. The abstract (minus its conclusion) answers the question
        t = table("qiaojin/PubMedQA", "pqa_artificial/train-00000-of-00001.parquet", ["question", "context"])
        return [(q, " ".join(c["contexts"])) for q, c in zip(t.column("question").to_pylist(),
                                                              t.column("context").to_pylist())]
    raise ValueError(source)


@app.function(image=image, cpu=8, memory=65536, volumes={"/vol": volume}, timeout=3 * 3600)
def build_mixed(per_source: dict, n_cand: int, pool_size: int, tag: str) -> dict:
    """v1 data: several sources, each query paired with its passage and n_cand-1 BM25 hard negatives mined
    from the same source's passages. Every passage known to be relevant to the query is excluded from its
    negatives (HotpotQA questions have two)."""
    import bm25s

    rng = random.Random(7)
    rows, stats = [], {}
    for source, n_queries in per_source.items():
        pairs = [(q.strip(), p.strip()) for q, p in _source_pairs(source) if q and p and len(p) > 20]
        relevant: dict[str, set[str]] = {}
        for q, p in pairs:
            relevant.setdefault(q, set()).add(p)
        questions = list(relevant)
        rng.shuffle(questions)
        corpus = list(dict.fromkeys(p for q in questions[:pool_size] for p in sorted(relevant[q])))
        picked = questions[: n_queries + n_queries // 100 + 10]  # a little spare for queries we have to skip
        retriever = bm25s.BM25()
        retriever.index(bm25s.tokenize(corpus, stopwords="en", show_progress=False), show_progress=False)
        hits, _ = retriever.retrieve(bm25s.tokenize(picked, stopwords="en", show_progress=False),
                                     k=n_cand + 8, show_progress=False, n_threads=8)
        kept = 0
        for q, ranked in zip(picked, hits.tolist()):
            if kept >= n_queries:
                break
            pos = sorted(relevant[q])[0]
            negatives = [corpus[i] for i in ranked if corpus[i] not in relevant[q]][: n_cand - 1]
            if len(negatives) < n_cand - 1:
                continue
            rows.append({"query": q, "docs": [pos] + negatives, "source": source})
            kept += 1
        stats[source] = {"queries": kept, "passages_in_pool": len(corpus), "pairs_available": len(pairs)}
        print(source, stats[source], flush=True)
    rng.shuffle(rows)
    held_n = min(300, len(rows) // 10)
    out = V / "data" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[held_n:]))
    (out / "heldout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:held_n]))
    volume.commit()
    return {"train_queries": len(rows) - held_n, "heldout": held_n, "candidates": n_cand, "sources": stats}


# -- teacher -------------------------------------------------------------------------


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=8 * 3600)
def teacher(tag: str) -> dict:
    import torch
    from common import Encoder, score_pairs

    device = torch.device("cuda")
    tok, model = _model("Qwen/Qwen3-Reranker-4B", device)
    enc = Encoder(tok)
    stats = {}
    for split in ("train", "heldout"):
        rows = [json.loads(x) for x in (V / "data" / tag / f"{split}.jsonl").read_text().splitlines()]
        pairs = [(r["query"], d) for r in rows for d in r["docs"]]
        t0 = time.time()
        # Score in shards saved to the Volume, so an interrupted run resumes instead of starting over.
        shard_dir = V / "data" / tag / f"{split}.teacher-shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard = 48000
        scores: list[float] = []
        for i in range(0, len(pairs), shard):
            f = shard_dir / f"{i:08d}.json"
            if f.exists():
                part = json.loads(f.read_text())
            else:
                part = score_pairs(torch, model, enc, pairs[i:i + shard], device, tokens_per_batch=24576)
                f.write_text(json.dumps(part))
                volume.commit()
            scores += part
            print(f"teacher {split}: {min(i + shard, len(pairs))}/{len(pairs)}", flush=True)
        k = 0
        for r in rows:
            r["teacher"] = scores[k:k + len(r["docs"])]
            k += len(r["docs"])
        (V / "data" / tag / f"{split}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        stats[split] = {"pairs": len(pairs), "seconds": round(time.time() - t0),
                        "positive_ranked_first": round(sum(r["teacher"].index(max(r["teacher"])) == 0 for r in rows) / len(rows), 3)}
        by_source: dict[str, list[int]] = {}
        for r in rows:  # how hard each source's negatives are (1.0 = the teacher always finds the positive)
            by_source.setdefault(r.get("source", "gooaq"), []).append(int(r["teacher"].index(max(r["teacher"])) == 0))
        stats[split]["positive_ranked_first_by_source"] = {s: round(sum(v) / len(v), 3) for s, v in by_source.items()}
    volume.commit()
    return stats


# -- train ---------------------------------------------------------------------------


@app.function(image=image, gpu="L40S", memory=32768, volumes={"/vol": volume}, timeout=10 * 3600)
def train(tag: str, epochs: float = 1.0, lr: float = 1e-5, queries_per_step: int = 8, max_steps: int = 0,
          name: str = NAME, ckpt_every: int = 500, halt_after: int = 0) -> dict:
    """Distil the teacher into the student. Saves a checkpoint (weights, optimiser, schedule, data order)
    every ckpt_every steps to the volume and resumes from it, so a timeout or a stopped app loses at most
    ckpt_every steps. halt_after stops early after a checkpoint (used only to test resuming)."""
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common import Encoder, forward_scores

    torch.manual_seed(0)
    device = torch.device("cuda")
    base = snapshot_download("Qwen/Qwen3-Reranker-0.6B", revision=REV["Qwen/Qwen3-Reranker-0.6B"],
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    enc = Encoder(tok)
    rows = [json.loads(x) for x in (V / "data" / tag / "train.scored.jsonl").read_text().splitlines()]
    held = [json.loads(x) for x in (V / "data" / tag / "heldout.scored.jsonl").read_text().splitlines()]
    flat = enc.encode_many([(r["query"], d) for r in rows + held for d in r["docs"]])
    k = 0
    for r in rows + held:
        r["ids"] = flat[k:k + len(r["docs"])]
        k += len(r["docs"])
    del flat

    steps = max_steps or int(math.ceil(len(rows) / queries_per_step) * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=True)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    def agreement(subset):
        model.eval()
        kl, top1 = [], 0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for r in subset:
                s = forward_scores(torch, model, enc, r["ids"], device)
                t = torch.tensor(r["teacher"], device=device)
                kl.append(float(F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="sum")))
                top1 += int(s.argmax() == t.argmax())
        model.train()
        return {"kl": round(sum(kl) / len(kl), 4), "top1_matches_teacher": round(top1 / len(subset), 3)}

    order = list(range(len(rows)))
    rng = random.Random(0)
    step, pos, log, done_before = 0, None, [], 0
    ck_dir = V / "ckpt" / name
    ck = ck_dir / "state.pt"
    cfg = {"tag": tag, "lr": lr, "steps": steps, "queries_per_step": queries_per_step, "train_queries": len(rows)}
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if state["cfg"] != cfg:
            raise RuntimeError(f"checkpoint {ck} was made with {state['cfg']}, this run is {cfg}. Move it away to start over.")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, pos, order, log, before = state["step"], state["pos"], state["order"], state["log"], state["before"]
        done_before = state["seconds"]
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        del state
        print(f"resumed from checkpoint at step {step} of {steps}", flush=True)
    else:
        before = agreement(held)
    print("held-out before:", before, flush=True)

    def checkpoint():
        ck_dir.mkdir(parents=True, exist_ok=True)
        tmp = ck_dir / "state.pt.tmp"
        torch.save({"cfg": cfg, "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "pos": pos, "order": order, "log": log, "before": before,
                    "seconds": done_before + round(time.time() - started), "rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state()}, tmp)
        tmp.replace(ck)
        volume.commit()
        print(f"checkpoint saved at step {step}", flush=True)

    model.train()
    started = time.time()
    while step < steps:
        if pos is None:  # a new pass over the data
            rng.shuffle(order)
            pos = 0
        while pos < len(order) and step < steps:
            batch = [rows[i] for i in order[pos:pos + queries_per_step]]
            pos += queries_per_step
            seqs = [ids for r in batch for ids in r["ids"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = forward_scores(torch, model, enc, seqs, device).view(len(batch), -1)
            t = torch.tensor([r["teacher"] for r in batch], device=device)
            listwise = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean")
            pointwise = F.binary_cross_entropy_with_logits(s, torch.sigmoid(t))
            loss = listwise + 0.5 * pointwise
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss.item(), 4), "listwise": round(listwise.item(), 4),
                            "lr": sched.get_last_lr()[0], "elapsed": done_before + round(time.time() - started)})
                print(log[-1], flush=True)
            if step < steps and ((ckpt_every and step % ckpt_every == 0) or step == halt_after):
                checkpoint()
                if step == halt_after:
                    return {"halted_at": step}
        if pos >= len(order):
            pos = None
    after = agreement(held)
    print("held-out after:", after, flush=True)
    out = V / "models" / name
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for f in Path(base).iterdir():  # keep the base model's own tokenizer files for converters
        if f.name.startswith(("tokenizer", "vocab", "merges", "special_tokens_map", "generation_config")):
            (out / f.name).write_bytes(f.read_bytes())
    summary = {"steps": step, "train_queries": len(rows), "seconds": done_before + round(time.time() - started), "lr": lr,
               "heldout_before": before, "heldout_after": after, "log": log}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    if ck.exists():  # finished, so the resume state is no longer needed (it is about 7 GB)
        ck.unlink()
    volume.commit()
    return summary


# -- eval ----------------------------------------------------------------------------


def _beir(name: str) -> tuple[dict, dict, dict]:
    import pyarrow.parquet as pq

    corpus_t = pq.read_table(_dl(f"BeIR/{name}", "corpus/corpus-00000-of-00001.parquet", "dataset"))
    queries_t = pq.read_table(_dl(f"BeIR/{name}", "queries/queries-00000-of-00001.parquet", "dataset"))
    qrels_path = _dl(f"BeIR/{name}-qrels", "test.tsv", "dataset")
    corpus = {i: (t + " " + x).strip() for i, t, x in zip(corpus_t.column("_id").to_pylist(),
                                                         corpus_t.column("title").to_pylist(),
                                                         corpus_t.column("text").to_pylist())}
    queries = dict(zip(queries_t.column("_id").to_pylist(), queries_t.column("text").to_pylist()))
    qrels: dict[str, dict[str, int]] = {}
    for line in Path(qrels_path).read_text().splitlines()[1:]:
        q, d, s = line.split("\t")
        qrels.setdefault(q, {})[d] = int(s)
    return corpus, queries, qrels


def _bm25_runs(name: str, k: int = 100) -> dict:
    cache = V / "beir" / f"{name}.bm25.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import bm25s

    corpus, queries, qrels = _beir(name)
    ids = list(corpus)
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([corpus[i] for i in ids], stopwords="en", show_progress=False), show_progress=False)
    qids = [q for q in qrels if q in queries]
    hits, _ = retriever.retrieve(bm25s.tokenize([queries[q] for q in qids], stopwords="en", show_progress=False),
                                 k=k + 1, show_progress=False)
    runs = {}
    for q, ranked in zip(qids, hits.tolist()):
        docs = [ids[i] for i in ranked if ids[i] != q][:k]  # ArguAna: never return the query's own argument
        runs[q] = docs
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(runs))
    volume.commit()
    return runs


@app.function(image=image, cpu=16, memory=98304, volumes={"/vol": volume}, timeout=6 * 3600)
def bm25(datasets: list[str]) -> dict:
    """BM25 top-100 for each BEIR test set, cached in the Volume (CPU only)."""
    out = {}
    for name in datasets:
        _, _, qrels = _beir(name)
        runs = _bm25_runs(name)
        out[name] = {"queries": len(runs), "bm25_ndcg10": round(sum(
            __import__("common").ndcg_at_10(r, qrels[q]) for q, r in runs.items()) / len(runs), 4)}
    return out


def _slim_path(name: str, max_queries: int) -> Path:
    return V / "beir" / f"{name}.eval-{max_queries or 'all'}.json"


@app.function(image=image, cpu=8, memory=65536, volumes={"/vol": volume}, timeout=4 * 3600)
def prep_eval(datasets: list[str], max_queries: int = 0) -> dict:
    """CPU only. For each BEIR set, keep just the sampled queries, their qrels and the documents BM25 put in
    their top 100, and cache that in the Volume. GPU workers then never load a full corpus (FEVER has 5.4M)."""
    out = {}
    for name in datasets:
        path = _slim_path(name, max_queries)
        if not path.exists():
            corpus, queries, qrels = _beir(name)
            runs = _bm25_runs(name)
            qids = list(runs)
            if max_queries and len(qids) > max_queries:  # a fixed random sample, the same for every model
                qids = sorted(random.Random(0).sample(qids, max_queries), key=qids.index)
            runs = {q: runs[q] for q in qids}
            needed = {d for docs in runs.values() for d in docs}
            slim = {"corpus": {d: corpus[d] for d in needed}, "queries": {q: queries[q] for q in qids},
                    "qrels": {q: qrels[q] for q in qids}, "runs": runs}
            del corpus
            path.write_text(json.dumps(slim))
            volume.commit()
        out[name] = round(path.stat().st_size / 1e6, 1)
        print(name, "ready", out[name], "MB", flush=True)
    return out


@app.function(image=image, gpu="L40S", memory=32768, volumes={"/vol": volume}, timeout=6 * 3600)
def evaluate(models: list[str], datasets: list[str], max_queries: int = 0) -> dict:
    """nDCG@10 of each model reranking BM25's top 100. Reads the sets prep_eval made, and saves each
    (model, set) result to the Volume as soon as it is done, so a rerun only scores what is missing."""
    import torch
    from common import Encoder, mean, ndcg_at_10, score_pairs

    volume.reload()
    device = torch.device("cuda")
    data = {name: json.loads(_slim_path(name, max_queries).read_text()) for name in datasets}
    results = {name: {"queries": len(d["runs"]),
                      "BM25": round(mean(ndcg_at_10(r, d["qrels"][q]) for q, r in d["runs"].items()), 4)}
               for name, d in data.items()}
    for m in models:
        label = m.split("/")[-1]
        cached = {name: V / "evals" / label / f"{name}.eval-{max_queries or 'all'}.json" for name in datasets}
        todo = [name for name in datasets if not cached[name].exists()]
        if todo:
            path = str(V / "models" / m) if (V / "models" / m).is_dir() else m
            tok, model = _model(path, device)
            enc = Encoder(tok)
            for name in todo:
                d = data[name]
                runs, qrels = d["runs"], d["qrels"]
                pairs = [(d["queries"][q], d["corpus"][doc]) for q, docs in runs.items() for doc in docs]
                t0 = time.time()
                scores = score_pairs(torch, model, enc, pairs, device)
                k, per_q = 0, []
                for q, docs in runs.items():
                    s = scores[k:k + len(docs)]
                    k += len(docs)
                    ranked = [doc for _, doc in sorted(zip(s, docs), key=lambda x: -x[0])]
                    per_q.append(ndcg_at_10(ranked, qrels[q]))
                row = {label: round(mean(per_q), 4), f"{label} pairs/s": round(len(pairs) / (time.time() - t0))}
                cached[name].parent.mkdir(parents=True, exist_ok=True)
                cached[name].write_text(json.dumps(row))
                volume.commit()
                print(name, label, row[label], flush=True)
            del model
            torch.cuda.empty_cache()
        for name in datasets:
            results[name].update(json.loads(cached[name].read_text()))
    return results


# -- export --------------------------------------------------------------------------


@app.function(image=export_image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3600)
def export(name: str = NAME) -> dict:
    src = V / "models" / name
    out = V / "gguf" / f"{name}-q8_0.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(src), "--outfile", str(out),
                    "--outtype", "q8_0"], check=True)
    subprocess.run(f"cd /tmp && echo '{RERANKD_SHA256_PINNED}  /opt/{RERANKD[1]}' | sha256sum -c - && "
                   f"tar -xzf /opt/{RERANKD[1]}", shell=True, check=True)
    binary = next(Path("/tmp").rglob("rerankd"))
    check = subprocess.run([str(binary), "check", str(out)], capture_output=True, text=True)
    volume.commit()
    return {"gguf": str(out), "bytes": out.stat().st_size, "rerankd_check": check.stdout + check.stderr,
            "ok": check.returncode == 0}


@app.local_entrypoint()
def main(stage: str = "all", smoke: bool = False, version: str = "v0", max_queries: int = 0, steps: int = 0) -> None:
    """version v0 reproduces the published run exactly. version v1 is the broader mix (see README).
    steps caps training steps (0 means one full epoch)."""
    if version not in NAMES:
        raise SystemExit(f"unknown version {version}")
    v1 = version == "v1"
    tag = ("smoke-" if smoke else "") + version if v1 else ("smoke" if smoke else "v0")
    name = NAMES[version] if not smoke else NAMES[version] + "-smoke"
    results = HERE / "results"
    results.mkdir(exist_ok=True)

    def save(what, obj):
        (results / f"{tag}-{what}.json").write_text(json.dumps(obj, indent=2))
        print(what, json.dumps({k: v for k, v in obj.items() if k != "log"} if isinstance(obj, dict) else obj, indent=2))

    if stage in ("data", "all"):
        if v1:
            per = {s: 40 for s in SOURCES_V1} if smoke else SOURCES_V1
            save("data", build_mixed.remote(per, 16, 5000 if smoke else 150000, tag))
        else:
            save("data", build_data.remote(200 if smoke else 12000, 16, 20000 if smoke else 300000, tag))
    if stage in ("teacher", "all"):
        save("teacher", teacher.remote(tag))
    if stage in ("train", "all"):
        lr = 5e-6 if v1 else 1e-5  # v1: gentler, to keep the base model's breadth (v0 lost ArguAna and science)
        save("train", train.remote(tag, lr=lr, max_steps=20 if smoke else steps, name=name))
    if stage in ("eval", "all"):
        sets = ["scifact"] if smoke else (BEIR_V1 if v1 else BEIR)
        limit = 30 if smoke else (max_queries or (1000 if v1 else 0))
        if not (results / f"{tag}-bm25.json").exists():
            save("bm25", bm25.remote(sets))
        print("prep_eval", prep_eval.remote(sets, limit))
        calls = [evaluate.spawn([m], sets, limit) for m in ("Qwen/Qwen3-Reranker-0.6B", name)]
        if not smoke and not v1:  # the 4B teacher for context, on the two smallest sets (it is 7x slower)
            calls.append(evaluate.spawn(["Qwen/Qwen3-Reranker-4B"], ["scifact", "nfcorpus"], 0))
        merged: dict = {}
        for c in calls:
            for ds, row in c.get().items():
                merged.setdefault(ds, {}).update(row)
        if v1:
            label = name.split("/")[-1]
            zs = [d for d in merged if d not in NOT_ZERO_SHOT]
            merged["_summary"] = {
                "zero_shot_sets": zs,
                "mean_zero_shot": {m: round(sum(merged[d][m] for d in zs) / len(zs), 4)
                                   for m in ("BM25", "Qwen3-Reranker-0.6B", label) if all(m in merged[d] for d in zs)},
                "not_zero_shot": sorted(NOT_ZERO_SHOT & set(merged)),
                "max_queries_per_set": limit,
            }
        save("eval", merged)
    if stage in ("export", "all"):
        save("export", export.remote(name))
