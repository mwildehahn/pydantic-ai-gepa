"""Tests for the GEPA configuration model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pydantic_ai_gepa.gepa_graph.models import (
    CandidateSelectorStrategy,
    GepaConfig,
)


def test_config_defaults() -> None:
    config = GepaConfig()
    assert config.max_evaluations == 200
    assert config.component_selector == "round_robin"
    assert config.candidate_selector is CandidateSelectorStrategy.PARETO
    assert config.validation_policy == "full"


def test_config_validation_errors() -> None:
    with pytest.raises(ValidationError):
        GepaConfig(max_evaluations=0)

    with pytest.raises(ValidationError):
        GepaConfig(minibatch_size=0)

    with pytest.raises(ValidationError):
        GepaConfig(max_iterations=0)
