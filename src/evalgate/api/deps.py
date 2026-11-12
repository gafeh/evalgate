from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from evalgate.db.session import session_scope


def get_engine(request: Request) -> AsyncEngine:
    engine: AsyncEngine = request.app.state.engine
    return engine


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with session_scope(get_engine(request)) as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]