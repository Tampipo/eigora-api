# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the QM router.
"""

import base64
import json
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from physense_api.main import app


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def client():
    return TestClient(app)


class TestEigenstates:
    async def test_harmonic_oscillator(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": 256},
                "potential": {"type": "harmonic", "params": {"omega": 1.0}},
                "n_states": 4,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["energies"]) == 4
        assert len(data["wavefunctions"]) == 4
        assert len(data["x"]) == 256

    async def test_energies_ascending(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": 256},
                "potential": {"type": "harmonic", "params": {"omega": 1.0}},
                "n_states": 4,
            })
        energies = resp.json()["energies"]
        assert all(energies[i] < energies[i+1] for i in range(len(energies)-1))

    async def test_harmonic_energy_values(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": 512},
                "potential": {"type": "harmonic", "params": {"omega": 1.0}},
                "n_states": 3,
            })
        energies = resp.json()["energies"]
        assert abs(energies[0] - 0.5) < 0.01
        assert abs(energies[1] - 1.5) < 0.01
        assert abs(energies[2] - 2.5) < 0.01

    async def test_barrier(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -10, "x_max": 10, "n_points": 256},
                "potential": {"type": "barrier", "params": {"height": 2.0, "width": 1.0}},
                "n_states": 4,
            })
        assert resp.status_code == 200

    async def test_custom_potential(self, async_client):
        n = 256
        x = np.linspace(-8, 8, n)
        V = (0.5 * x**2).tolist()
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": n},
                "potential": {"type": "custom", "values": V},
                "n_states": 3,
            })
        assert resp.status_code == 200
        energies = resp.json()["energies"]
        assert abs(energies[0] - 0.5) < 0.05

    async def test_custom_wrong_size(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": 256},
                "potential": {"type": "custom", "values": [0.0] * 100},
                "n_states": 3,
            })
        assert resp.status_code == 422

    async def test_invalid_grid(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": 5, "x_max": -5, "n_points": 256},
                "potential": {"type": "free"},
                "n_states": 3,
            })
        assert resp.status_code == 422

    async def test_n_states_too_large(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/eigenstates", json={
                "grid": {"x_min": -8, "x_max": 8, "n_points": 256},
                "potential": {"type": "harmonic"},
                "n_states": 100,
            })
        assert resp.status_code == 422


class TestSingleAtomState:
    async def test_hydrogen_1s(self, async_client):
        # 1s is nodeless and everywhere positive: a single +ψ lobe, no −ψ lobe.
        async with async_client as c:
            resp = await c.post("/qm/single-atom-state", json={
                "grid": {"x_min": -10, "x_max": 10, "y_min": -10, "y_max": 10,
                         "z_min": -10, "z_max": 10, "nx": 32, "ny": 32, "nz": 32},
                "Z": 1, "n": 1, "l": 0, "m": 0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["bound_radius"] > 0
        assert data["negative"] is None
        lobe = data["positive"]
        assert lobe is not None
        assert lobe["vertex_count"] > 0
        assert lobe["triangle_count"] > 0
        # positions are base64 of flat float32 [x,y,z,...] — 3 floats/vertex.
        assert len(base64.b64decode(lobe["positions"])) == lobe["vertex_count"] * 3 * 4
        assert len(base64.b64decode(lobe["indices"])) == lobe["triangle_count"] * 3 * 4

    async def test_both_lobes_for_p_orbital(self, async_client):
        # 2p_z has a node at theta=pi/2, so ψ takes both signs -> two lobes.
        async with async_client as c:
            resp = await c.post("/qm/single-atom-state", json={
                "grid": {"x_min": -10, "x_max": 10, "y_min": -10, "y_max": 10,
                         "z_min": -10, "z_max": 10, "nx": 32, "ny": 32, "nz": 32},
                "Z": 1, "n": 2, "l": 1, "m": 0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["positive"] is not None
        assert data["negative"] is not None
        assert data["positive"]["triangle_count"] > 0
        assert data["negative"]["triangle_count"] > 0

    async def test_invalid_quantum_numbers(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/single-atom-state", json={
                "grid": {"nx": 32, "ny": 32, "nz": 32},
                "Z": 1, "n": 1, "l": 1, "m": 0,  # l must be < n
            })
        assert resp.status_code == 422

    async def test_grid_resolution_too_low(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/single-atom-state", json={
                "grid": {"nx": 8, "ny": 8, "nz": 8},
                "Z": 1, "n": 1, "l": 0, "m": 0,
            })
        assert resp.status_code == 422


class TestEvolveWebSocket:
    def test_evolve_metadata_and_frames(self, client):
        with client.websocket_connect("/qm/evolve") as ws:
            ws.send_text(json.dumps({
                "grid": {"x_min": -20, "x_max": 20, "n_points": 256},
                "potential": {"type": "barrier", "params": {"height": 2.0, "width": 2.0}},
                "wavepacket": {"x0": -8.0, "k0": 1.5, "sigma": 1.5},
                "t_max": 3.0,
                "dt": 0.01,
                "n_frames": 10,
            }))

            # First message: metadata
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "metadata"
            assert "x" in msg
            assert "potential" in msg

            # Collect frames
            frames = []
            while True:
                msg = json.loads(ws.receive_text())
                if msg["type"] == "done":
                    break
                assert msg["type"] == "frame"
                frames.append(msg)

            assert len(frames) >= 10
            assert all("probability_density" in f for f in frames)
            assert all(abs(f["norm"] - 1.0) < 0.05 for f in frames)

    def test_predicted_transmission_for_barrier(self, client):
        with client.websocket_connect("/qm/evolve") as ws:
            ws.send_text(json.dumps({
                "grid": {"x_min": -20, "x_max": 20, "n_points": 256},
                "potential": {"type": "barrier", "params": {"height": 5.0, "width": 1.0}},
                "wavepacket": {"x0": -8.0, "k0": 2.0, "sigma": 1.0},
                "t_max": 1.0,
                "dt": 0.01,
                "n_frames": 10,
            }))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "metadata"
            assert msg["predicted_transmission"] is not None
            assert 0.0 < msg["predicted_transmission"] < 1.0
            assert msg["mean_energy_transmission"] is not None
            assert 0.0 < msg["mean_energy_transmission"] < 1.0
            # T(E) is convex in the tunnelling regime, so the energy-averaged
            # prediction should exceed T at a single mean energy (Jensen's
            # inequality) -- see physense_qm's scattering tests.
            assert msg["predicted_transmission"] > msg["mean_energy_transmission"]

            # Drain the rest of the stream
            while json.loads(ws.receive_text())["type"] != "done":
                pass

    def test_predicted_transmission_none_for_non_barrier(self, client):
        with client.websocket_connect("/qm/evolve") as ws:
            ws.send_text(json.dumps({
                "grid": {"x_min": -20, "x_max": 20, "n_points": 256},
                "potential": {"type": "harmonic", "params": {"omega": 1.0}},
                "wavepacket": {"x0": -8.0, "k0": 1.5, "sigma": 1.5},
                "t_max": 1.0,
                "dt": 0.01,
                "n_frames": 10,
            }))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "metadata"
            assert msg["predicted_transmission"] is None
            assert msg["mean_energy_transmission"] is None

            while json.loads(ws.receive_text())["type"] != "done":
                pass
