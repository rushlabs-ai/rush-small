"""Rush Small: distil Qwen3-Reranker-4B into Qwen3-Reranker-0.6B with permissively licensed data.

    modal run pipeline.py --stage all --smoke     # tiny end-to-end check, a few cents
    modal run pipeline.py --stage all             # the real run
    modal run pipeline.py --stage eval            # re-run one stage (data|teacher|train|eval|export)

Stages, all cached in the Modal Volume "rush-small":
  data     GooAQ (Apache-2.0) questions; candidates = the true answer + BM25 hard negatives
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
}
BEIR = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs"]
NAME = "rush-small-qwen3-reranker-0.6b-v0"  # "qwen3-reranker" lets llama.cpp detect the format
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


# -- teacher -------------------------------------------------------------------------


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=3 * 3600)
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
        scores = score_pairs(torch, model, enc, pairs, device, tokens_per_batch=24576,
                             progress=lambda d, n: print(f"teacher {split}: {d}/{n}", flush=True))
        k = 0
        for r in rows:
            r["teacher"] = scores[k:k + len(r["docs"])]
            k += len(r["docs"])
        (V / "data" / tag / f"{split}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        stats[split] = {"pairs": len(pairs), "seconds": round(time.time() - t0),
                        "positive_ranked_first": round(sum(r["teacher"].index(max(r["teacher"])) == 0 for r in rows) / len(rows), 3)}
    volume.commit()
    return stats


# -- train ---------------------------------------------------------------------------


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=4 * 3600)
def train(tag: str, epochs: float = 1.0, lr: float = 1e-5, queries_per_step: int = 8, max_steps: int = 0) -> dict:
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
    for r in rows + held:
        r["ids"] = [enc.encode(r["query"], d) for d in r["docs"]]

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

    before = agreement(held)
    print("held-out before:", before, flush=True)
    log = []
    model.train()
    order = list(range(len(rows)))
    rng = random.Random(0)
    started = time.time()
    step = 0
    while step < steps:
        rng.shuffle(order)
        for b in range(0, len(order), queries_per_step):
            batch = [rows[i] for i in order[b:b + queries_per_step]]
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
                log.append({"step": step, "loss": round(float(loss), 4), "listwise": round(float(listwise), 4),
                            "lr": sched.get_last_lr()[0], "elapsed": round(time.time() - started)})
                print(log[-1], flush=True)
            if step >= steps:
                break
    after = agreement(held)
    print("held-out after:", after, flush=True)
    out = V / "models" / NAME
    model.config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for f in Path(base).iterdir():  # keep the base model's own tokenizer files for converters
        if f.name.startswith(("tokenizer", "vocab", "merges", "special_tokens_map", "generation_config")):
            (out / f.name).write_bytes(f.read_bytes())
    summary = {"steps": step, "train_queries": len(rows), "seconds": round(time.time() - started), "lr": lr,
               "heldout_before": before, "heldout_after": after, "log": log}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
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


@app.function(image=image, cpu=8, memory=32768, volumes={"/vol": volume}, timeout=2 * 3600)
def bm25(datasets: list[str]) -> dict:
    """BM25 top-100 for each BEIR test set, cached in the Volume (CPU only)."""
    out = {}
    for name in datasets:
        _, _, qrels = _beir(name)
        runs = _bm25_runs(name)
        out[name] = {"queries": len(runs), "bm25_ndcg10": round(sum(
            __import__("common").ndcg_at_10(r, qrels[q]) for q, r in runs.items()) / len(runs), 4)}
    return out


@app.function(image=image, gpu="L40S", volumes={"/vol": volume}, timeout=4 * 3600)
def evaluate(models: list[str], datasets: list[str], max_queries: int = 0) -> dict:
    import torch
    from common import Encoder, mean, ndcg_at_10, score_pairs

    volume.reload()
    device = torch.device("cuda")
    data = {}
    for name in datasets:
        corpus, queries, qrels = _beir(name)
        runs = _bm25_runs(name)
        qids = list(runs)[: max_queries or None]
        data[name] = (corpus, queries, qrels, {q: runs[q] for q in qids})
    results = {name: {"queries": len(d[3]), "BM25": round(mean(ndcg_at_10(r, d[2][q]) for q, r in d[3].items()), 4)}
               for name, d in data.items()}
    for m in models:
        path = str(V / "models" / m) if (V / "models" / m).is_dir() else m
        tok, model = _model(path, device)
        enc = Encoder(tok)
        label = m.split("/")[-1]
        for name, (corpus, queries, qrels, runs) in data.items():
            pairs = [(queries[q], corpus[d]) for q, docs in runs.items() for d in docs]
            t0 = time.time()
            scores = score_pairs(torch, model, enc, pairs, device)
            k, per_q = 0, []
            for q, docs in runs.items():
                s = scores[k:k + len(docs)]
                k += len(docs)
                ranked = [d for _, d in sorted(zip(s, docs), key=lambda x: -x[0])]
                per_q.append(ndcg_at_10(ranked, qrels[q]))
            results[name][label] = round(mean(per_q), 4)
            results[name][f"{label} pairs/s"] = round(len(pairs) / (time.time() - t0))
            print(name, label, results[name][label], flush=True)
        del model
        torch.cuda.empty_cache()
    return results


# -- export --------------------------------------------------------------------------


@app.function(image=export_image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=3600)
def export() -> dict:
    src = V / "models" / NAME
    out = V / "gguf" / f"{NAME}-q8_0.gguf"
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
def main(stage: str = "all", smoke: bool = False) -> None:
    tag = "smoke" if smoke else "v0"
    results = HERE / "results"
    results.mkdir(exist_ok=True)

    def save(name, obj):
        (results / f"{tag}-{name}.json").write_text(json.dumps(obj, indent=2))
        print(name, json.dumps({k: v for k, v in obj.items() if k != "log"} if isinstance(obj, dict) else obj, indent=2))

    if stage in ("data", "all"):
        save("data", build_data.remote(200 if smoke else 12000, 16, 20000 if smoke else 300000, tag))
    if stage in ("teacher", "all"):
        save("teacher", teacher.remote(tag))
    if stage in ("train", "all"):
        save("train", train.remote(tag, max_steps=20 if smoke else 0))
    if stage in ("eval", "all"):
        sets = ["scifact"] if smoke else BEIR
        limit = 30 if smoke else 0
        save("bm25", bm25.remote(sets))
        calls = [evaluate.spawn([m], sets, limit) for m in ("Qwen/Qwen3-Reranker-0.6B", NAME)]
        if not smoke:  # the 4B teacher for context, on the two smallest sets (it is 7x slower)
            calls.append(evaluate.spawn(["Qwen/Qwen3-Reranker-4B"], ["scifact", "nfcorpus"], 0))
        merged: dict = {}
        for c in calls:
            for name, row in c.get().items():
                merged.setdefault(name, {}).update(row)
        save("eval", merged)
    if stage in ("export", "all"):
        save("export", export.remote())
