from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from evalgate import __version__
from evalgate.api.routers import datasets, reports, review, runs, ui
from evalgate.config import get_settings
from evalgate.db.session import build_engine

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def create_app(
    engine: AsyncEngine | None = None, *, read_only: bool | None = None
) -> FastAPI:
    """Build the app. Pass an engine to bind it to a specific database."""
    locked = get_settings().read_only if read_only is None else read_only

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
    app.state.read_only = locked

    if locked:

        @app.middleware("http")
        async def _reject_writes(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            if request.method not in SAFE_METHODS:
                return JSONResponse(
                    {"detail": "this deployment is read-only"},
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                )
            return await call_next(request)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str | bool]:
        return {"status": "ok", "version": __version__, "read_only": locked}

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    app.include_router(datasets.router)
    app.include_router(runs.router)
    app.include_router(review.router)
    app.include_router(reports.router)
    app.include_router(ui.router)
    return app


app = create_app()