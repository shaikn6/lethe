# lethe

A memory server for AI agents that learns which memories to return.

An agent with long-term memory has to pick a handful of stored items to read before it answers. Most memory systems pick with a fixed recipe: some mix of keyword match and embedding similarity, perhaps with a bonus for recent items. lethe starts from the same recipe and then adjusts it from feedback. After each recall the agent says whether the memories helped, and a small reinforcement-learning policy shifts the ranking towards what works.

It is one Go binary with a journal on disk, an HTTP API, and adapters for Python, LangChain and MCP.

## What has been measured

**Does the learning help?** On [LoCoMo](https://github.com/snap-research/locomo) (ten long conversations, 5,882 stored turns, 1,535 questions with labelled evidence), the score is how much of the evidence is among the returned memories. The learned ranking was trained with feedback on eight conversations and measured, with learning switched off, on the two it had never seen; all five such splits, five seeds.

| Ranking                                    | Evidence in top 5 | Evidence in top 10 |
| ------------------------------------------ | ----------------- | ------------------ |
| Keyword only                               | 0.438             | 0.512              |
| Embedding similarity only                  | 0.479             | 0.585              |
| Equal mix of both                          | 0.493             | 0.570              |
| Mix + neighbouring turns, hand-set weights | 0.545             | 0.641              |
| Learned from one score per recall          | 0.568 ± 0.006     | 0.658 ± 0.004      |
| **Learned from "which memories helped"**   | **0.618 ± 0.002** | **0.707 ± 0.003**  |

**Does that hold on other data?** Only partly. On [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (470 questions, each with its own chat history of about 500 turns; three seeds), the learned ranking ties the best hand-set rule and does not beat it.

| Ranking | Evidence in top 5 | Evidence in top 10 |
|---|---|---|
| Keyword only | 0.681 | 0.789 |
| Embedding similarity only | 0.707 | 0.845 |
| Equal mix of both | 0.727 | 0.832 |
| Mix + neighbouring turns, hand-set weights | 0.687 | 0.817 |
| Learned from "which memories helped" | 0.729 ± 0.021 | 0.847 ± 0.022 |

What the two tables show together is narrower than "learning wins". The recipe that was best by hand on LoCoMo (neighbours) is the worst mix on LongMemEval, and the other way round. The learned ranking ended level with the best hand-set rule on one dataset and ahead of it on the other without being told which recipe suited which data. On LongMemEval it also varies more between seeds than the fixed rules do, with only 376 training questions per split.

**Does it survive a crash?** The server was killed with `kill -9` in the middle of 32 clients writing, three times. Every memory it had acknowledged was there after restart (9,725, 11,110 and 11,862 of them), and the journal was intact.

**Does it survive bad input?** A fuzz test sent 540,000 random request bodies at the API: no crash, no server error.

**How fast is it?** On an Apple M2 Pro, 32 clients each looping store, recall, feedback for ten seconds: 7,119 requests per second, no failed requests, half of the three-request loops done within 12.5 ms and 99% within 31 ms. That is one laptop, loopback networking and 64-number embeddings, not a capacity claim.

## Read this before quoting a number

- **At test time the learned ranking is a fixed set of weights.** The result says that weights learned from feedback on some users' conversations carry over to other users and beat the weights I set by hand. It does not say that learning beats every rule someone could write, and anyone can now copy the learned weights.
- **The feedback in the benchmark is simulated.** A real agent would report whether a recall helped. Here that report is computed from the benchmark's evidence labels: the share of evidence returned, or which returned memories were evidence. The held-out conversations never give feedback.
- **An earlier version of the learner lost** to the hand-set rule (0.611 against 0.641 in the top 10). The cause was that memories next to a strong match start with a score of zero, so the policy never tried them and never learned they were useful. Adding a small amount of forced exploration fixed it. I mention it because the first number I got was the wrong one to believe.
- **One learned signal is untested here.** The policy also tracks how often each memory has helped before. Within one user's history that should help repeated questions; on conversations it has never seen it is constant, so this benchmark says nothing about it.
- **The retrieval score is not answer quality.** No language model was asked to answer anything.

## Run it

Requires Go 1.26.

```sh
go build -o bin/lethed ./cmd/lethed
LETHE_TOKEN=change-me bin/lethed -journal lethe.journal        # listens on 127.0.0.1:8787
```

```sh
# store, recall, say what helped
curl -s localhost:8787/v1/memories -H "Authorization: Bearer change-me" \
  -d '{"namespace":"alice","text":"Priya is allergic to peanuts"}'
curl -s localhost:8787/v1/recall -H "Authorization: Bearer change-me" \
  -d '{"namespace":"alice","query":"what can Priya not eat","explore":true}'
curl -s localhost:8787/v1/feedback -H "Authorization: Bearer change-me" \
  -d '{"recall_id":"<from the recall>","reward":1,"useful":["<memory id>"]}'
```

The server does not compute embeddings. Send one with each memory and query from whatever model you use, or send none and it ranks by keywords and neighbours alone.

From Python, LangChain or an MCP client:

```python
from lethe import Lethe
from lethe.langchain import LetheRetriever

memory = Lethe("http://127.0.0.1:8787", namespace="alice", token="change-me", embed=my_embedder)
memory.remember("Priya is allergic to peanuts")
recall = memory.recall("what can Priya not eat")
memory.feedback(recall.id, 1.0, useful=[recall.memories[0].id])

retriever = LetheRetriever(client=memory, k=5)  # documents carry recall_id and memory_id
```

```sh
go build -o bin/lethe-mcp ./cmd/lethe-mcp
LETHE_TOKEN=change-me bin/lethe-mcp -namespace alice    # remember, recall, feedback, forget over stdio
```

To repeat the checks:

```sh
go test -race ./...
go test ./internal/server -run xxx -fuzz FuzzAPI -fuzztime 30s
python -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
bench/.venv/bin/python bench/locomo.py          # needs data/locomo10.json from the LoCoMo repository
```

## How it works

**Candidates.** A query gathers the best keyword matches (BM25) and the best embedding matches, plus the memories stored right next to the strongest of them. That last part matters in conversations: the turn that answers a question often shares no words with it but sits directly after one that does. In LoCoMo, adding neighbours within two turns would lift the evidence found from 57% to 78% if ranked perfectly.

**Ranking.** Each candidate is described by nine numbers: keyword score, embedding similarity, recency, how often it helped before, importance, recency for questions about time, whether both searches found it, and how well its two neighbours match by keyword and by embedding. The policy scores candidates as a weighted sum and, while learning, draws the slate at random in proportion to those scores, with a small chance per slot of a uniformly random pick.

**Learning.** Feedback moves the weights by policy gradient (REINFORCE with Adam). The reward can be one number for the whole recall or, better, the ids of the memories that helped, the way an agent that cites its sources would report it. Per-memory feedback is much less noisy, which is why it learns a better ranking above.

**Storage.** Every change is written to an append-only journal before it is acknowledged, and replayed on start. A write that fails part-way is rolled back so a half-written record can never sit in the middle of the file; if even that fails, the server refuses further writes rather than risk the journal. Feedback and its journal record succeed or fail together. Forgetting a memory removes its text from RAM at once and from disk at the next compaction, which runs after a thousand forgets, on shutdown and on start.

**Around it.** A bounded, non-blocking event stream can publish what the store does to Kafka; a slow or absent broker costs events, never requests, and events carry ids and outcomes only. `deploy/terraform` runs the server on one AWS instance with the journal on its own encrypted volume.

## Limits

- **Namespaces separate data, not tenants.** Anyone holding the API token can read and write every namespace.
- **Embedding search is brute force**, fine up to roughly a hundred thousand memories per namespace and not beyond.
- **The Kafka path has only met a stand-in broker** in tests, and the Terraform has been validated but never applied.
- **One server, one journal.** There is no replication.

## Licence

MIT. See `LICENSE`.
