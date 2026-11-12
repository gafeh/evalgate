from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from evalgate.api.deps import SessionDep
from evalgate.api.schemas import DatasetOut
from evalgate.core.dataset import Dataset as CoreDataset
from evalgate.db.repositories import (
    DatasetConflictError,
    get_dataset,
    list_datasets,
    save_dataset,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetOut])
async def index(session: SessionDep, limit: int = 100) -> list[DatasetOut]:
    rows = await list_datasets(session, limit=limit)
    return [DatasetOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/{name}/{version}", response_model=DatasetOut)
async def show(name: str, version: str, session: SessionDep) -> DatasetOut:
    row = await get_dataset(session, name, version)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no dataset {name}@{version}")
    return DatasetOut.model_validate(row, from_attributes=True)


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def create(payload: CoreDataset, session: SessionDep) -> DatasetOut:
    """Store a dataset. Idempotent for identical content, 409 on a changed one."""
    try:
        row = await save_dataset(session, payload)
    except DatasetConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return DatasetOut.model_validate(row, from_attributes=True)