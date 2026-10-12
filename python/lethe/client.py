"""HTTP client for the lethe memory server."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Self

import httpx

Embedder = Callable[[str], Sequence[float]]


@dataclass(frozen=True)
class RecalledMemory:
    """One memory returned for a query."""

    id: str
    text: str
    score: float
    importance: float
    created_at: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Recall:
    """The answer to one query. Pass ``id`` to ``Lethe.feedback``."""

    id: str
    memories: list[RecalledMemory]


class Lethe:
    """Client for one namespace of a lethe server.

    ``embed`` turns text into a vector with whatever model you choose; the
    server only stores and compares the vectors. Without it, recall falls
    back to keyword search.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        *,
        namespace: str = "default",
        token: str | None = None,
        embed: Embedder | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Create a client; no request is made until a method is called."""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.namespace = namespace
        self._embed = embed
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    def close(self) -> None:
        """Release the HTTP connection."""
        self._http.close()

    def __enter__(self) -> Self:
        """Return the client."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the client when the block ends."""
        self.close()

    def remember(
        self,
        text: str,
        *,
        importance: float | None = None,
        metadata: dict[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> str:
        """Store a memory and return its id."""
        body: dict = {"namespace": self.namespace, "text": text}
        if self._embed:
            body["embedding"] = list(self._embed(text))
        if importance is not None:
            body["importance"] = importance
        if metadata:
            body["metadata"] = metadata
        if created_at is not None:
            body["created_at"] = created_at.isoformat()
        return self._post("/v1/memories", body)["id"]

    def recall(self, query: str, *, k: int = 5, explore: bool = True) -> Recall:
        """Return the memories most likely to help with ``query``.

        Keep ``explore`` on while you send feedback: the server then tries
        slightly different rankings so it can learn which work. Turn it off
        for a fixed, repeatable ranking.
        """
        body: dict = {"namespace": self.namespace, "query": query, "k": k, "explore": explore}
        if self._embed:
            body["embedding"] = list(self._embed(query))
        out = self._post("/v1/recall", body)
        memories = [
            RecalledMemory(
                id=m["id"],
                text=m["text"],
                score=m["score"],
                importance=m["importance"],
                created_at=m["created_at"],
                metadata=m.get("metadata") or {},
            )
            for m in out["memories"]
        ]
        return Recall(id=out["recall_id"], memories=memories)

    def feedback(self, recall_id: str, reward: float, *, useful: Sequence[str] | None = None) -> None:
        """Tell the server how much a recall helped, from 0 to 1.

        ``useful`` names the returned memories that were actually used, for
        example the ones the agent cited. It makes learning much faster.
        """
        body: dict = {"recall_id": recall_id, "reward": reward}
        if useful is not None:
            body["useful"] = list(useful)
        self._post("/v1/feedback", body)

    def forget(self, memory_id: str) -> None:
        """Delete a memory."""
        self._http.delete(f"/v1/memories/{self.namespace}/{memory_id}").raise_for_status()

    def _post(self, path: str, body: dict) -> dict:
        response = self._http.post(path, json=body)
        response.raise_for_status()
        return response.json() if response.content else {}
