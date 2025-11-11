"""Tests for GEPA graph construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, cast

from pydantic_graph import Graph

from pydantic_ai_gepa.adapter import Adapter
from pydantic_ai_gepa.gepa_graph.graph import create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import GepaConfig
from pydantic_ai_gepa.gepa_graph.nodes import (
    ContinueNode,
    EvaluateNode,
    MergeNode,
    ReflectNode,
    StartNode,
)
from pydantic_ai_gepa.types import DataInst


@dataclass
class _AgentStub:
    name: str | None = None


@dataclass
class _AdapterStub:
    agent: _AgentStub
    input_spec: None = None

    async def evaluate(self, batch, candidate, capture_traces):  # pragma: no cover - unused
        raise RuntimeError("evaluate should not be called in this test")

    def make_reflective_dataset(
        self,
        *,
        candidate,
        eval_batch,
        components_to_update: Sequence[str],
    ) -> dict[str, list[dict]]:  # pragma: no cover - unused
        return {component: [] for component in components_to_update}

    def get_components(self) -> dict[str, str]:  # pragma: no cover - unused
        return {"instructions": "seed"}


def _make_adapter(name: str | None = None) -> Adapter[DataInst]:
    return cast(Adapter[DataInst], _AdapterStub(agent=_AgentStub(name=name)))


def test_create_gepa_graph_without_merge() -> None:
    adapter = _make_adapter()
    config = GepaConfig()

    graph = create_gepa_graph(adapter=adapter, config=config)

    assert isinstance(graph, Graph)
    assert graph.name == "gepa_graph"
    assert set(graph.node_defs) == {
        "StartNode",
        "EvaluateNode",
        "ContinueNode",
        "ReflectNode",
        "MergeNode",
    }
    assert graph.node_defs["StartNode"].node is StartNode
    assert graph.node_defs["ContinueNode"].node is ContinueNode


def test_create_gepa_graph_with_merge_enabled() -> None:
    adapter = _make_adapter()
    config = GepaConfig(use_merge=True)

    graph = create_gepa_graph(adapter=adapter, config=config)

    assert isinstance(graph, Graph)
    assert graph.node_defs["MergeNode"].node is MergeNode
