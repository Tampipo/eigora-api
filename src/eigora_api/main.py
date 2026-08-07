# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Physense API — FastAPI application.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from physense_api.routers import qm


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Physense API",
        description="Physics simulation backend for the Physense platform.",
        version="0.1.0",
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

    app.include_router(qm.router)

    return app


app = create_app()
