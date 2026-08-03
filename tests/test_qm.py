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


class TestSeparableState:
    # A square 2D box, wide enough that both axes are the same well.
    SQUARE_BOX = {"type": "infinite_well", "params": {"width": 4.0, "x0": 0.0}}
    HARMONIC = {"type": "harmonic", "params": {"omega": 1.0}}

    @staticmethod
    def body(**overrides):
        request = {
            "grid": {"x_min": -4, "x_max": 4, "y_min": -4, "y_max": 4,
                     "nx": 128, "ny": 128},
            "potential_x": TestSeparableState.SQUARE_BOX,
            "potential_y": TestSeparableState.SQUARE_BOX,
        }
        request.update(overrides)
        return request

    async def test_box_in_two_directions(self, async_client):
        # A well of width 6 along x and 3 along y, third state along x and
        # fourth along y: 3 lobes across, 4 up.
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json={
                "grid": {"x_min": 0, "x_max": 6, "y_min": 0, "y_max": 3,
                         "nx": 128, "ny": 128},
                "potential_x": {"type": "infinite_well",
                                "params": {"width": 6.0, "x0": 3.0}},
                "potential_y": {"type": "infinite_well",
                                "params": {"width": 3.0, "x0": 1.5}},
                "n1": 2, "n2": 3,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == [3, 4]          # box quantum numbers start at 1
        assert data["quantum_numbers"] == ["n_1", "n_2"]
        assert data["is_exact"] is True
        assert len(data["x"]) == 128
        assert len(data["psi_x"]) == 128
        # E = n1^2 pi^2 / 2 L_x^2 + n2^2 pi^2 / 2 L_y^2
        assert data["energy_x"] == pytest.approx(9 * np.pi**2 / (2 * 36))
        assert data["energy_y"] == pytest.approx(16 * np.pi**2 / (2 * 9))
        assert data["energy"] == pytest.approx(data["energy_x"] + data["energy_y"])

    async def test_factors_reconstruct_a_normalised_field(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(n1=1, n2=2))
        data = resp.json()
        psi = np.outer(data["psi_x"], data["psi_y"])
        dx = data["x"][1] - data["x"][0]
        dy = data["y"][1] - data["y"][0]
        assert np.sum(psi**2) * dx * dy == pytest.approx(1.0, abs=1e-3)

    async def test_ground_state_has_no_nodes(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(n1=0, n2=0))
        data = resp.json()
        inside = np.array(data["psi_x"])[np.abs(data["x"]) < 1.9]
        assert np.all(inside > 0) or np.all(inside < 0)

    async def test_square_box_degeneracy(self, async_client):
        async with async_client as c:
            degenerate = await c.post("/qm/separable-state", json=self.body(n1=0, n2=1))
            ground = await c.post("/qm/separable-state", json=self.body(n1=0, n2=0))
        assert degenerate.json()["degeneracy"] == 2   # (1,2) and (2,1)
        assert ground.json()["degeneracy"] == 1

    async def test_isotropic_trap_degeneracy(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x=self.HARMONIC, potential_y=self.HARMONIC, n1=1, n2=1,
            ))
        data = resp.json()
        assert data["energy"] == pytest.approx(3.0)   # (1+1/2) + (1+1/2)
        assert data["degeneracy"] == 3                # level n=2 of a 2D trap

    async def test_axes_may_differ(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x=self.HARMONIC,
                potential_y={"type": "harmonic", "params": {"omega": 2.0}},
                n1=0, n2=0,
            ))
        data = resp.json()
        assert data["energy"] == pytest.approx(0.5 + 1.0)
        assert data["degeneracy"] == 1

    async def test_numerical_axis_is_not_exact(self, async_client):
        # A finite well has no analytic spectrum, so that axis is solved on the grid.
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x=self.HARMONIC,
                potential_y={"type": "finite_well",
                             "params": {"depth": 10.0, "width": 3.0}},
                n1=0, n2=1, n_states=5,
            ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_exact"] is False
        assert data["label"] == [0, 1]   # numerical states are 0-indexed

    async def test_state_index_beyond_what_was_computed(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x=self.HARMONIC,
                potential_y={"type": "finite_well",
                             "params": {"depth": 10.0, "width": 3.0}},
                n1=0, n2=40, n_states=5,
            ))
        assert resp.status_code == 422
        assert "out of range" in resp.json()["detail"]

    async def test_custom_potential_on_one_axis(self, async_client):
        # Option B: a custom V(x) that happens to be a harmonic well, so the
        # numerically solved axis must reproduce the analytic one on y.
        values = (0.5 * np.linspace(-4, 4, 128) ** 2).tolist()
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x={"type": "custom", "values": values},
                potential_y=self.HARMONIC,
                n1=1, n2=0, n_states=6,
            ))
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_exact"] is False
        assert data["energy_x"] == pytest.approx(1.5, abs=1e-2)
        assert data["energy_y"] == pytest.approx(0.5)

    async def test_custom_potential_size_is_validated(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                potential_x={"type": "custom", "values": [0.0] * 7},
            ))
        assert resp.status_code == 422

    async def test_invalid_grid(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(
                grid={"x_min": 4, "x_max": -4, "y_min": -4, "y_max": 4,
                      "nx": 128, "ny": 128},
            ))
        assert resp.status_code == 422

    async def test_negative_state_index(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/separable-state", json=self.body(n1=-1))
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
