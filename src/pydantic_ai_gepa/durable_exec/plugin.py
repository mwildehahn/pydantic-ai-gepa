"""Worker configuration helpers for GEPA optimization."""

from __future__ import annotations

from typing import Any, Sequence

from temporalio.worker import Plugin as WorkerPlugin, Worker, WorkerConfig

from pydantic_ai_gepa.durable_exec.activities import (
    load_dataset_ids,
    load_dataset_batch,
    calculate_metric,
    reflect_candidate,
)
from pydantic_ai_gepa.durable_exec.workflow import GepaOptimizationWorkflow

# Exported lists so users can compose/extend.
GEPA_WORKFLOWS = [GepaOptimizationWorkflow]
GEPA_ACTIVITIES = [
    load_dataset_ids,
    load_dataset_batch,
    calculate_metric,
    reflect_candidate,
]


class GepaPlugin(WorkerPlugin):
    """Temporal worker plugin that registers GEPA workflows/activities.

    Optionally, you can ask it to also register a TemporalAgent's activities
    (if you are *not* using pydantic-ai's AgentPlugin). By default it leaves
    agent registration to AgentPlugin to avoid duplicate activity names.
    """

    def __init__(
        self,
        *,
        workflows: Sequence[Any] | None = None,
        activities: Sequence[Any] | None = None,
        temporal_agent: Any | None = None,
    ) -> None:
        self.workflows = (
            list(workflows) if workflows is not None else list(GEPA_WORKFLOWS)
        )
        self.activities = (
            list(activities) if activities is not None else list(GEPA_ACTIVITIES)
        )
        self.temporal_agent = temporal_agent

    def init_worker_plugin(self, next: WorkerPlugin) -> None:
        self.next = next

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        config["workflows"] = [*config.get("workflows", []), *self.workflows]

        activity_list = [*config.get("activities", []), *self.activities]
        if self.temporal_agent is not None:
            activity_list.extend(self.temporal_agent.temporal_activities)
        config["activities"] = activity_list

        return self.next.configure_worker(config)

    async def run_worker(self, worker: Worker) -> None:
        await self.next.run_worker(worker)
