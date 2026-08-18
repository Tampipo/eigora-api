# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Eigora API — FastAPI application.
"""

import os
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eigora_api.routers import qm

try:
    __version__ = version("eigora-api")
except PackageNotFoundError:
    # Not installed as a package (e.g. running straight from source) --
    # release-please only ever bumps pyproject.toml, so there's no second
    # copy of the version to fall back on here.
    __version__ = "0.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Eigora API",
        description="Physics simulation backend for the Eigora platform.",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS — configured via environment variable for prod (k3s)
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(qm.router, prefix="/v1")

    return app


app = create_app()
