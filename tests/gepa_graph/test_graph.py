"""Tests for GEPA graph construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic_graph import Graph

from pydantic_ai_gepa.adapter import PydanticAIGEPAAdapter
from pydantic_ai_gepa.gepa_graph.graph import create_gepa_graph
from pydantic_ai_gepa.gepa_graph.models import GepaConfig
from pydantic_ai_gepa.gepa_graph.nodes import (
    ContinueNode,
    EvaluateNode,
    MergeNode,
    ReflectNode,
    StartNode,
)


@dataclass
class _AgentStub:
    name: str | None = None


@dataclass
class _AdapterStub:
    agent: _AgentStub


def _make_adapter(name: str | None = None) -> PydanticAIGEPAAdapter[Any]:
    return cast(PydanticAIGEPAAdapter[Any], _AdapterStub(agent=_AgentStub(name=name)))


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
    }
    assert "MergeNode" not in graph.node_defs
    assert graph.node_defs["StartNode"].node is StartNode
    assert graph.node_defs["ContinueNode"].node is ContinueNode


def test_create_gepa_graph_with_merge_enabled() -> None:
    adapter = _make_adapter()
    config = GepaConfig(use_merge=True)

    graph = create_gepa_graph(adapter=adapter, config=config)

    assert isinstance(graph, Graph)
    assert "MergeNode" in graph.node_defs
    assert graph.node_defs["MergeNode"].node is MergeNode
    assert set(graph.node_defs) == {
        "StartNode",
        "EvaluateNode",
        "ContinueNode",
        "ReflectNode",
        "MergeNode",
    }
