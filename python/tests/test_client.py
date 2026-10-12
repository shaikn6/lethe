"""Run the client and the LangChain retriever against a real lethed."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from lethe import Lethe

BINARY = Path(__file__).resolve().parents[2] / "bin" / "lethed"


def toy_embed(text: str) -> list[float]:
    """Embed by counting a few topic words; enough to tell topics apart."""
    words = text.lower().split()
    return [
        1.0 + sum(w.startswith(("cat", "kitten", "feline")) for w in words),
        1.0 + sum(w.startswith(("train", "rail", "station")) for w in words),
    ]


@pytest.fixture
def server_url():
    if not BINARY.exists():
        pytest.skip("build the server first: go build -o bin/lethed ./cmd/lethed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [str(BINARY), "-addr", f"127.0.0.1:{port}", "-journal", "", "-seed", "7"],
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(url + "/healthz")
            break
        except httpx.TransportError:
            time.sleep(0.05)
    yield url
    process.terminate()
    process.wait()


def test_remember_recall_feedback_forget(server_url):
    with Lethe(server_url, namespace="alice", embed=toy_embed) as memory:
        cat = memory.remember("The kitten is called Miso", metadata={"source": "chat"})
        memory.remember("The train to Leeds leaves from the old station")

        recall = memory.recall("what is my feline called", k=1, explore=False)
        assert [m.id for m in recall.memories] == [cat]
        assert recall.memories[0].metadata == {"source": "chat"}

        memory.feedback(recall.id, 1.0, useful=[cat])
        with pytest.raises(httpx.HTTPStatusError):
            memory.feedback(recall.id, 1.0)

        memory.forget(cat)
        assert cat not in [m.id for m in memory.recall("kitten Miso", explore=False).memories]


def test_namespaces_do_not_leak(server_url):
    with Lethe(server_url, namespace="alice") as alice, Lethe(server_url, namespace="bob") as bob:
        alice.remember("alice keeps her spare key under the blue pot")
        assert bob.recall("spare key", explore=False).memories == []


def test_langchain_retriever_returns_documents_that_can_be_rated(server_url):
    pytest.importorskip("langchain_core")
    from lethe.langchain import LetheRetriever

    with Lethe(server_url, namespace="docs", embed=toy_embed) as memory:
        memory.remember("Railcards cut train fares by a third", metadata={"topic": "travel"})
        memory.remember("Cats sleep for most of the day")

        retriever = LetheRetriever(client=memory, k=1, explore=False)
        docs = retriever.invoke("cheaper train tickets")

        assert len(docs) == 1
        assert docs[0].page_content.startswith("Railcards")
        assert docs[0].metadata["topic"] == "travel"
        memory.feedback(docs[0].metadata["recall_id"], 1.0, useful=[docs[0].metadata["memory_id"]])


def test_a_crafted_id_cannot_reach_another_namespace(server_url):
    with Lethe(server_url, namespace="alice") as alice, Lethe(server_url, namespace="bob") as bob:
        secret = alice.remember("alice keeps her passport in the red folder")
        for crafted in (f"../alice/{secret}", f"..%2Falice%2F{secret}", f"x/../../alice/{secret}"):
            with pytest.raises(httpx.HTTPStatusError):
                bob.forget(crafted)
        assert [m.id for m in alice.recall("passport folder", explore=False).memories] == [secret]
