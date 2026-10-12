"""Benchmark lethe on LoCoMo: does a ranking learned from reward beat fixed rules.

Every dialogue turn of the ten LoCoMo conversations is stored as one memory.
For each question the server returns its top memories, and the score is how
many of the labelled evidence turns are among them.

Fixed rankings are measured on all ten conversations. The learned ranking is
cross-validated: it is trained with feedback on eight conversations and
measured, with learning switched off, on the two it has never seen.

During training the reward is the share of evidence turns in the returned
slate. That stands in for a user saying "this helped"; the held-out
conversations never give feedback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Self

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
KS = (5, 10)

FEATURES = [
    "keyword",
    "semantic",
    "recency",
    "utility",
    "importance",
    "temporal-recency",
    "agreement",
    "neighbour-keyword",
    "neighbour-semantic",
]
FIXED = {
    "keyword only": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "semantic only": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "hybrid (equal mix)": [2, 2, 0, 0, 0, 0, 0, 0, 0],
    "hybrid + recency": [2, 2, 1, 0, 0, 0, 0, 0, 0],
    "hybrid + neighbours": [2, 2, 0, 0, 0, 0, 0, 1, 1],
}
ADMIN_TOKEN = "bench-admin"
START = FIXED["hybrid (equal mix)"]


def load(path: Path) -> list[dict]:
    """Return conversations as {id, turns, questions, as_of}."""
    conversations = []
    for sample in json.loads(path.read_text()):
        conv = sample["conversation"]
        turns, last = [], None
        for key in sorted(
            (k for k in conv if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        ):
            when = datetime.strptime(
                conv[f"{key}_date_time"], "%I:%M %p on %d %B, %Y"
            ).replace(tzinfo=timezone.utc)
            for offset, turn in enumerate(conv[key]):
                text = f"{turn['speaker']}: {turn['text']}"
                if turn.get("blip_caption"):
                    text += f" [shares a photo: {turn['blip_caption']}]"
                turns.append(
                    {
                        "dia_id": turn["dia_id"],
                        "text": text,
                        "created_at": when + timedelta(seconds=offset),
                    }
                )
                last = when
        known = {t["dia_id"] for t in turns}
        questions = []
        for qa in sample["qa"]:
            evidence = {
                e
                for raw in qa.get("evidence", [])
                for e in re.findall(r"D\d+:\d+", raw)
            } & known
            if qa.get("category") in CATEGORIES and evidence:
                questions.append(
                    {
                        "question": qa["question"],
                        "evidence": evidence,
                        "category": qa["category"],
                    }
                )
        conversations.append(
            {
                "id": sample["sample_id"],
                "turns": turns,
                "questions": questions,
                "as_of": last + timedelta(days=1),
            }
        )
    return conversations


def embed(
    texts: list[str], model_name: str, cache: Path, prefix: str = ""
) -> np.ndarray:
    """Embed texts with a local open model, caching by content."""
    digest = hashlib.sha256(
        ("\n".join([model_name, prefix, *texts])).encode()
    ).hexdigest()[:16]
    file = cache / f"emb_{digest}.npy"
    if file.exists():
        return np.load(file)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [prefix + t for t in texts],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(file, vectors)
    return vectors


class Server:
    """A lethed process with an in-memory store, for one experiment."""

    def __init__(self, seed: int, learning_rate: float, temperature: float):
        """Start lethed on a free port and wait until it answers."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.process = subprocess.Popen(
            [
                str(ROOT / "bin" / "lethed"),
                "-addr",
                f"127.0.0.1:{port}",
                "-journal",
                "",
                "-seed",
                str(seed),
                "-learning-rate",
                str(learning_rate),
                "-temperature",
                str(temperature),
            ],
            stderr=subprocess.DEVNULL,
            env={**os.environ, "LETHE_ADMIN_TOKEN": ADMIN_TOKEN},
        )
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
        for _ in range(100):
            try:
                self.http.get("/healthz")
                return
            except httpx.TransportError:
                time.sleep(0.05)
        raise RuntimeError("lethed did not start")

    def close(self) -> None:
        """Stop the server process."""
        self.http.close()
        self.process.terminate()
        self.process.wait()

    def __enter__(self) -> Self:
        """Return the running server."""
        return self

    def __exit__(self, *exc) -> None:
        """Stop the server when the block ends."""
        self.close()

    def post(self, path: str, body: dict) -> dict:
        """POST JSON and return the decoded reply, failing on any HTTP error."""
        response = self.http.post(path, json=body)
        response.raise_for_status()
        return response.json() if response.content else {}

    def set_weights(self, weights: list[float]) -> None:
        """Pin the ranking weights."""
        self.http.put(
            "/v1/policy",
            json={"weights": weights},
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        ).raise_for_status()

    def weights(self) -> list[float]:
        """Return the current ranking weights."""
        return self.http.get("/healthz").json()["weights"]

    def ingest(self, conversations: list[dict]) -> None:
        """Store every turn and remember which memory holds which turn.

        Conversations are loaded in parallel, but the turns of each one go in
        order, because the server treats memories stored next to each other
        as neighbours.
        """

        def load_one(conv: dict) -> None:
            conv["memory_to_dia"] = {}
            for turn in conv["turns"]:
                stored = self.post(
                    "/v1/memories",
                    {
                        "namespace": conv["id"],
                        "text": turn["text"],
                        "embedding": turn["embedding"],
                        "created_at": turn["created_at"].isoformat(),
                    },
                )
                conv["memory_to_dia"][stored["id"]] = turn["dia_id"]
            conv["dia_to_memory"] = {d: m for m, d in conv["memory_to_dia"].items()}

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(load_one, conversations))

    def recall(
        self, conv: dict, question: dict, k: int, explore: bool
    ) -> tuple[str, list[str]]:
        """Return the recall id and the dialogue ids of the returned memories."""
        out = self.post(
            "/v1/recall",
            {
                "namespace": conv["id"],
                "query": question["question"],
                "embedding": question["embedding"],
                "k": k,
                "as_of": conv["as_of"].isoformat(),
                "explore": explore,
            },
        )
        return out["recall_id"], [
            conv["memory_to_dia"][m["id"]] for m in out["memories"]
        ]


def evaluate(server: Server, conversations: list[dict]) -> list[dict]:
    """Greedy recall for every question; one row per question.

    When a question lists evidence sessions, the row also says what share of
    those sessions had at least one turn returned.
    """
    rows = []
    for conv in conversations:
        for question in conv["questions"]:
            _, found = server.recall(conv, question, max(KS), explore=False)
            row = {"category": question["category"]}
            for k in KS:
                got = question["evidence"] & set(found[:k])
                row[f"recall@{k}"] = len(got) / len(question["evidence"])
                row[f"hit@{k}"] = float(bool(got))
                sessions = question.get("evidence_sessions")
                if sessions:
                    seen = {d.rsplit(":", 1)[0] for d in found[:k]}
                    row[f"session@{k}"] = len(sessions & seen) / len(sessions)
            rows.append(row)
    return rows


def train(
    server: Server,
    conversations: list[dict],
    epochs: int,
    k: int,
    rng: random.Random,
    per_memory: bool,
) -> None:
    """Run feedback-driven training.

    The slate reward is the share of evidence returned. With per_memory the
    feedback also names which returned memories were evidence, the way an
    agent that cites its sources would.
    """
    pairs = [(conv, q) for conv in conversations for q in conv["questions"]]
    for _ in range(epochs):
        rng.shuffle(pairs)
        for conv, question in pairs:
            recall_id, found = server.recall(conv, question, k, explore=True)
            hit = question["evidence"] & set(found)
            body = {
                "recall_id": recall_id,
                "reward": len(hit) / len(question["evidence"]),
            }
            if per_memory:
                body["useful"] = [conv["dia_to_memory"][d] for d in hit]
            server.post("/v1/feedback", body)


def summarize(rows: list[dict], categories: dict | None = None) -> dict:
    """Average each metric overall and per question category."""
    categories = CATEGORIES if categories is None else categories
    out = {"questions": len(rows)}
    for metric in sorted({key for row in rows for key in row} - {"category"}):
        out[metric] = statistics.fmean(r[metric] for r in rows)
        for cat, name in categories.items():
            subset = [r[metric] for r in rows if r["category"] == cat]
            if subset:
                out[f"{metric} {name}"] = statistics.fmean(subset)
    return out


def main() -> None:
    """Run the fixed rankings and the cross-validated learned ranking."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "locomo10.json")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-k", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    conversations = load(args.data)
    cache = args.data.parent
    turn_vectors = embed(
        [t["text"] for c in conversations for t in c["turns"]], args.model, cache
    )
    question_vectors = embed(
        [q["question"] for c in conversations for q in c["questions"]],
        args.model,
        cache,
        QUERY_PREFIX,
    )
    turns = iter(turn_vectors.tolist())
    questions = iter(question_vectors.tolist())
    for conv in conversations:
        for turn in conv["turns"]:
            turn["embedding"] = next(turns)
        for question in conv["questions"]:
            question["embedding"] = next(questions)
    print(
        f"{len(conversations)} conversations, {len(turn_vectors)} memories, "
        f"{len(question_vectors)} questions, embeddings from {args.model}"
    )

    results: dict = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "fixed": {},
        "learned": {},
    }

    with Server(1, args.learning_rate, args.temperature) as server:
        server.ingest(conversations)
        for name, weights in FIXED.items():
            server.set_weights(weights)
            results["fixed"][name] = summarize(evaluate(server, conversations))

    folds = [conversations[i : i + 2] for i in range(0, len(conversations), 2)]
    modes = (("learned, slate reward", False), ("learned, per-memory reward", True))
    for mode, per_memory in modes:
        per_seed, learned_weights = [], []
        for seed in range(1, args.seeds + 1):
            rows = []
            for held_out in folds:
                training = [c for c in conversations if c not in held_out]
                with Server(seed, args.learning_rate, args.temperature) as server:
                    server.ingest(conversations)
                    server.set_weights(START)
                    train(
                        server,
                        training,
                        args.epochs,
                        args.train_k,
                        random.Random(seed),
                        per_memory,
                    )
                    learned_weights.append(server.weights())
                    rows += evaluate(server, held_out)
            per_seed.append(summarize(rows))
            print(
                f"{mode} seed {seed}: recall@5 {per_seed[-1]['recall@5']:.4f}  "
                f"recall@10 {per_seed[-1]['recall@10']:.4f}",
                flush=True,
            )
        learned = {"questions": per_seed[0]["questions"]}
        for metric in per_seed[0]:
            if metric != "questions":
                values = [row[metric] for row in per_seed]
                learned[metric] = statistics.fmean(values)
                learned[f"{metric} std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        learned["weights_mean"] = np.mean(learned_weights, axis=0).round(3).tolist()
        learned["weights_std"] = np.std(learned_weights, axis=0).round(3).tolist()
        results["learned"][mode] = learned

    args.out.mkdir(exist_ok=True)
    out_file = args.out / f"locomo_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(results, indent=2))

    print(
        f"\n{'ranking':<30}{'recall@5':>10}{'recall@10':>11}{'hit@5':>8}{'hit@10':>8}"
    )
    for name, row in results["fixed"].items():
        print(
            f"{name:<30}{row['recall@5']:>10.4f}{row['recall@10']:>11.4f}{row['hit@5']:>8.4f}{row['hit@10']:>8.4f}"
        )
    for name, row in results["learned"].items():
        print(
            f"{name:<30}{row['recall@5']:>10.4f}{row['recall@10']:>11.4f}{row['hit@5']:>8.4f}{row['hit@10']:>8.4f}"
            f"   (± {row['recall@5 std']:.4f} / {row['recall@10 std']:.4f} over {args.seeds} seeds, held-out)"
        )
    for name, row in results["learned"].items():
        print(f"weights after training, {name}:")
        pairs = zip(FEATURES, row["weights_mean"], row["weights_std"])
        print("  " + ", ".join(f"{f} {m:.2f}±{d:.2f}" for f, m, d in pairs))
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
