"""Dataset loader utilities for the GEPA graph implementation."""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Hashable,
    Protocol,
    Sequence,
    TypeVar,
    TypeAlias,
    runtime_checkable,
    cast,
)

from pydantic_evals import Case, Dataset


class ComparableHashable(Hashable, Protocol):
    """Protocol requiring hashing plus rich comparison support."""

    def __lt__(self, other: Any, /) -> bool: ...

    def __gt__(self, other: Any, /) -> bool: ...

    def __le__(self, other: Any, /) -> bool: ...

    def __ge__(self, other: Any, /) -> bool: ...


DataIdT = TypeVar("DataIdT", bound=ComparableHashable)
CaseT = TypeVar("CaseT", bound=Case[Any, Any, Any])

DatasetPayload: TypeAlias = (
    Sequence[Case[Any, Any, Any]] | "DataLoader[Any, Case[Any, Any, Any]]"
)
DatasetFactory: TypeAlias = Callable[[], DatasetPayload | Awaitable[DatasetPayload]]
DatasetInput: TypeAlias = (
    DatasetPayload
    | DatasetFactory
    | Awaitable[DatasetPayload]
    | "DatasetLoader[Case[Any, Any, Any]]"
    | Dataset[Any, Any, Any]
)


def data_id_for_instance(case: Case[Any, Any, Any], index: int) -> str:
    """Return the identifier used for evaluation bookkeeping."""

    return case.name or f"case-{index}"


@runtime_checkable
class DataLoader(Protocol[DataIdT, CaseT]):
    """Interface for retrieving dataset instances by opaque identifiers."""

    async def all_ids(self) -> Sequence[DataIdT]: ...

    async def fetch(self, ids: Sequence[DataIdT]) -> list[CaseT]: ...

    def __len__(self) -> int: ...


class MutableDataLoader(DataLoader[DataIdT, CaseT], Protocol):
    """Data loader variant that supports appending new items."""

    async def add_items(self, items: Sequence[CaseT]) -> None: ...


class ListDataLoader(MutableDataLoader[ComparableHashable, Case[Any, Any, Any]]):
    """In-memory loader backed by a concrete list of instances."""

    def __init__(self, items: Sequence[Case[Any, Any, Any]]) -> None:
        self._items = list(items)
        self._ids: list[ComparableHashable] = []
        self._index_by_id: dict[ComparableHashable, int] = {}
        self._rebuild_index()

    async def all_ids(self) -> Sequence[ComparableHashable]:
        return list(self._ids)

    async def fetch(
        self, ids: Sequence[ComparableHashable]
    ) -> list[Case[Any, Any, Any]]:
        batch: list[Case[Any, Any, Any]] = []
        for data_id in ids:
            idx = self._index_by_id.get(data_id)
            if idx is None:
                raise KeyError(
                    f"Unknown data id {data_id!r} requested from ListDataLoader."
                )
            batch.append(self._items[idx])
        return batch

    def __len__(self) -> int:
        return len(self._items)

    async def add_items(self, items: Sequence[Case[Any, Any, Any]]) -> None:
        for item in items:
            self._append_item(item)

    def _rebuild_index(self) -> None:
        self._ids = []
        self._index_by_id = {}
        for idx, item in enumerate(self._items):
            self._append_item(item, explicit_index=idx)

    def _append_item(
        self,
        item: Case[Any, Any, Any],
        *,
        explicit_index: int | None = None,
    ) -> None:
        idx = explicit_index if explicit_index is not None else len(self._items)
        if explicit_index is None:
            self._items.append(item)
        data_id = data_id_for_instance(item, idx)
        if data_id in self._index_by_id:
            raise ValueError(
                f"Duplicate data id {data_id!r} detected in ListDataLoader."
            )
        if explicit_index is None:
            self._ids.append(data_id)
        else:
            if idx < len(self._ids):
                self._ids[idx] = data_id
            else:
                self._ids.append(data_id)
        self._index_by_id[data_id] = idx


@runtime_checkable
class DatasetLoader(Protocol[CaseT]):
    """Async factory that materializes a dataset on demand."""

    async def load(self) -> Sequence[CaseT] | DataLoader[Any, CaseT]: ...


def ensure_loader(
    data_or_loader: DatasetPayload,
) -> DataLoader[Any, Case[Any, Any, Any]]:
    """Return a DataLoader regardless of whether a sequence or loader was provided."""

    if isinstance(data_or_loader, DataLoader):
        return cast(DataLoader[Any, Case[Any, Any, Any]], data_or_loader)
    if _is_sequence_like(data_or_loader):
        return ListDataLoader(cast(Sequence[Case[Any, Any, Any]], data_or_loader))
    raise TypeError(f"Unable to coerce {type(data_or_loader)!r} into a DataLoader.")


async def resolve_dataset(
    dataset: DatasetInput,
    *,
    name: str,
) -> DataLoader[Any, Case[Any, Any, Any]]:
    """Materialize ``dataset`` into a concrete :class:`DataLoader`."""

    resolved = await _materialize_dataset(dataset, name=name)
    loader = ensure_loader(resolved)
    if len(loader) == 0:
        raise ValueError(f"{name} must contain at least one instance.")
    return loader


async def _materialize_dataset(
    dataset: DatasetInput,
    *,
    name: str,
) -> DatasetPayload:
    if dataset is None:
        raise ValueError(f"{name} is required.")

    if isinstance(dataset, DataLoader):
        return cast(DataLoader[Any, Case[Any, Any, Any]], dataset)

    if isinstance(dataset, DatasetLoader):
        return await dataset.load()

    if isinstance(dataset, Dataset):
        return cast(Sequence[Case[Any, Any, Any]], dataset.cases)

    if _is_sequence_like(dataset):
        return cast(Sequence[Case[Any, Any, Any]], dataset)

    if inspect.isawaitable(dataset):
        awaited = await cast(Awaitable[DatasetPayload], dataset)
        return await _materialize_dataset(awaited, name=name)

    if callable(dataset):
        produced = dataset()
        if inspect.isawaitable(produced):
            produced = await cast(Awaitable[DatasetPayload], produced)
        return await _materialize_dataset(produced, name=name)

    raise TypeError(
        f"{name} must be a sequence, DataLoader, or awaitable factory. Got {type(dataset)!r}."
    )


def _is_sequence_like(value: object) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, Sequence)


__all__ = [
    "ComparableHashable",
    "DataLoader",
    "DatasetInput",
    "DatasetLoader",
    "ListDataLoader",
    "MutableDataLoader",
    "data_id_for_instance",
    "ensure_loader",
    "resolve_dataset",
]
