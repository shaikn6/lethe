"""LangChain retriever backed by lethe.

Each returned document carries ``memory_id`` and ``recall_id`` in its
metadata, so the chain or graph that used it can report back with
``Lethe.feedback`` and the ranking improves over time.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from lethe.client import Lethe


class LetheRetriever(BaseRetriever):
    """Retrieve memories from a lethe server."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Lethe
    k: int = 5
    explore: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        recall = self.client.recall(query, k=self.k, explore=self.explore)
        return [
            Document(
                page_content=m.text,
                metadata={
                    **m.metadata,
                    "memory_id": m.id,
                    "recall_id": recall.id,
                    "score": m.score,
                    "created_at": m.created_at,
                },
            )
            for m in recall.memories
        ]
