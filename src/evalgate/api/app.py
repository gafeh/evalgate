from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate import __version__
from evalgate.api.routers import datasets, reports, review, runs
from evalgate.db.session import build_engine


def create_app(engine: AsyncEngine | None = None) -> FastAPI:
    """Build the app. Pass an engine to bind it to a specific database."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = engine is None
        app.state.engine = engine or build_engine()
        try:
            yield
        finally:
            if owned:
                await app.state.engine.dispose()

    app = FastAPI(
        title="EvalGate",
        version=__version__,
        summary="LLM evaluation harness with judge calibration and regression gating",
        lifespan=lifespan,
    )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(datasets.router)
    app.include_router(runs.router)
    app.include_router(review.router)
    app.include_router(reports.router)
    return app


app = create_app()