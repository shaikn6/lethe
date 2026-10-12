"""Benchmark lethe on LongMemEval, a second and independent dataset.

Each of the 500 questions comes with its own chat history of about 48
sessions. Every turn of that history is stored as one memory in a namespace
of its own, and the score is how many of the turns that hold the answer are
returned. Questions that are meant to be unanswerable are left out, because
they have no evidence to find.

The learned ranking is cross-validated over questions: trained with
feedback on four fifths of the histories and measured on the rest.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import locomo as base
import numpy as np

MAX_CHARS = 2000  # the embedding model reads about this much of a turn
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


def load(path: Path) -> list[dict]:
    """Return one history per question, shaped like a LoCoMo conversation."""
    histories = []
    for item in json.loads(path.read_text()):
        turns, evidence = [], set()
        for session_id, date, session in zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
        ):
            when = datetime.strptime(date, DATE_FORMAT).replace(tzinfo=timezone.utc)
            for offset, turn in enumerate(session):
                dia_id = f"{session_id}:{offset}"
                turns.append(
                    {
                        "dia_id": dia_id,
                        "text": f"{turn['role']}: {turn['content']}"[:MAX_CHARS],
                        "created_at": when + timedelta(seconds=offset),
                    }
                )
                if turn.get("has_answer"):
                    evidence.add(dia_id)
        if not evidence:
            continue
        asked = datetime.strptime(item["question_date"], DATE_FORMAT).replace(
            tzinfo=timezone.utc
        )
        histories.append(
            {
                "id": item["question_id"],
                "turns": turns,
                "as_of": asked,
                "questions": [
                    {
                        "question": item["question"],
                        "evidence": evidence,
                        "evidence_sessions": {d.rsplit(":", 1)[0] for d in evidence},
                        "category": item["question_type"],
                    }
                ],
            }
        )
    return histories


def attach_embeddings(histories: list[dict], model: str, cache: Path) -> None:
    """Embed each distinct turn once; histories share many sessions."""
    texts = sorted({t["text"] for h in histories for t in h["turns"]})
    vectors = dict(zip(texts, base.embed(texts, model, cache).tolist()))
    questions = [q["question"] for h in histories for q in h["questions"]]
    question_vectors = iter(
        base.embed(questions, model, cache, base.QUERY_PREFIX).tolist()
    )
    for history in histories:
        for turn in history["turns"]:
            turn["embedding"] = vectors[turn["text"]]
        for question in history["questions"]:
            question["embedding"] = next(question_vectors)
    print(
        f"{len(histories)} histories, {sum(len(h['turns']) for h in histories)} memories, {len(texts)} distinct turns"
    )


def main() -> None:
    """Run the fixed rankings and the cross-validated learned ranking."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data", type=Path, default=base.ROOT / "data" / "longmemeval_s.json"
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-k", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--per-memory", action="store_true", help="feedback names the useful memories"
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="compute and cache embeddings, then stop",
    )
    parser.add_argument("--out", type=Path, default=base.ROOT / "runs")
    args = parser.parse_args()

    histories = load(args.data)
    attach_embeddings(histories, args.model, args.data.parent)
    if args.embed_only:
        return
    categories = {
        c: c for c in sorted({h["questions"][0]["category"] for h in histories})
    }
    results: dict = {"args": {k: str(v) for k, v in vars(args).items()}, "fixed": {}}

    with base.Server(1, args.learning_rate, args.temperature) as server:
        server.ingest(histories)
        for name, weights in base.FIXED.items():
            server.set_weights(weights)
            results["fixed"][name] = base.summarize(
                base.evaluate(server, histories), categories
            )
            print(
                f"{name}: recall@10 {results['fixed'][name]['recall@10']:.4f}",
                flush=True,
            )

    order = list(range(len(histories)))
    random.Random(0).shuffle(order)
    folds = [[histories[i] for i in order[f::5]] for f in range(5)]
    per_seed, learned_weights = [], []
    for seed in range(1, args.seeds + 1):
        rows = []
        for held_out in folds:
            held = {h["id"] for h in held_out}
            training = [h for h in histories if h["id"] not in held]
            with base.Server(seed, args.learning_rate, args.temperature) as server:
                server.ingest(histories)
                server.set_weights(base.START)
                base.train(
                    server,
                    training,
                    args.epochs,
                    args.train_k,
                    random.Random(seed),
                    args.per_memory,
                )
                learned_weights.append(server.weights())
                rows += base.evaluate(server, held_out)
        per_seed.append(base.summarize(rows, categories))
        print(
            f"learned seed {seed}: recall@5 {per_seed[-1]['recall@5']:.4f}  recall@10 {per_seed[-1]['recall@10']:.4f}",
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
    results["learned"] = learned

    args.out.mkdir(exist_ok=True)
    out_file = args.out / f"longmemeval_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    out_file.write_text(json.dumps(results, indent=2))

    print(
        f"\n{'ranking':<26}{'recall@5':>10}{'recall@10':>11}{'session@5':>11}{'session@10':>12}"
    )
    rows = [*results["fixed"].items(), ("learned (held-out)", learned)]
    for name, row in rows:
        print(
            f"{name:<26}{row['recall@5']:>10.4f}{row['recall@10']:>11.4f}{row['session@5']:>11.4f}{row['session@10']:>12.4f}"
        )
    print(
        f"learned std over {args.seeds} seeds: recall@5 {learned['recall@5 std']:.4f}, recall@10 {learned['recall@10 std']:.4f}"
    )
    pairs = zip(base.FEATURES, learned["weights_mean"], learned["weights_std"])
    print("weights: " + ", ".join(f"{f} {m:.2f}±{d:.2f}" for f, m, d in pairs))
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
