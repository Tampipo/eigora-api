# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

from importlib.metadata import version

from eigora_api.main import app


def test_app_version_matches_package_metadata():
    """
    The FastAPI app's version comes from package metadata (pyproject.toml),
    not a hardcoded string -- this guards against the two drifting apart
    again, the way they had before.
    """
    assert app.version == version("eigora-api")
