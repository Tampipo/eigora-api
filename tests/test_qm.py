# Copyright (C) 2026 Tanguy Marsault - Eigora
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
from eigora_api.main import app


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


class TestDiscreteMeasurement:
    SIGMA_Z = {"re": [[1, 0], [0, -1]]}
    SIGMA_Y = {"re": [[0, 0], [0, 0]], "im": [[0, -1], [1, 0]]}

    async def test_plus_state_in_z_basis(self, async_client):
        # |+> = (|0> + |1>)/sqrt(2) is an even superposition of the two
        # sigma_z eigenstates, so both outcomes carry probability 1/2.
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 1]},
                "operator": self.SIGMA_Z,
            })
        assert resp.status_code == 200
        outcomes = resp.json()["outcomes"]
        assert [o["value"] for o in outcomes] == [-1.0, 1.0]
        assert all(abs(o["probability"] - 0.5) < 1e-12 for o in outcomes)
        assert all(o["degeneracy"] == 1 for o in outcomes)
        # Collapse lands on a basis state, up to a global phase.
        assert [abs(z) for z in outcomes[1]["state"]["re"]] == pytest.approx([1, 0])

    async def test_eigenstate_gives_certain_outcome(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0]},
                "operator": self.SIGMA_Z,
            })
        outcomes = resp.json()["outcomes"]
        assert outcomes[1]["probability"] == pytest.approx(1.0)
        # The impossible outcome has no collapsed state to report.
        assert outcomes[0]["probability"] == pytest.approx(0.0)
        assert outcomes[0]["state"] is None

    async def test_complex_amplitudes_are_not_dropped(self, async_client):
        # (|0> + i|1>)/sqrt(2) is the +1 eigenstate of sigma_y, so the outcome
        # is certain. Ignoring the imaginary part would give 1/2 - 1/2 instead.
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0], "im": [0, 1]},
                "operator": self.SIGMA_Y,
            })
        assert resp.status_code == 200
        outcomes = resp.json()["outcomes"]
        assert outcomes[1]["value"] == pytest.approx(1.0)
        assert outcomes[1]["probability"] == pytest.approx(1.0)

    async def test_degenerate_eigenvalue_is_one_outcome(self, async_client):
        # diag(1, 1, 2): the eigenvalue 1 spans a 2D subspace, so it appears
        # once with the probability summed over it, and the collapse projects
        # onto the whole subspace rather than onto one basis vector.
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 1, 1]},
                "operator": {"re": [[1, 0, 0], [0, 1, 0], [0, 0, 2]]},
            })
        outcomes = resp.json()["outcomes"]
        assert len(outcomes) == 2
        assert outcomes[0]["value"] == pytest.approx(1.0)
        assert outcomes[0]["degeneracy"] == 2
        assert outcomes[0]["probability"] == pytest.approx(2 / 3)
        collapsed = np.array(outcomes[0]["state"]["re"]) + 1j * np.array(
            outcomes[0]["state"]["im"]
        )
        assert np.abs(collapsed) == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    async def test_draws_are_counted_and_reproducible(self, async_client):
        body = {
            "state": {"re": [1, 1]},
            "operator": self.SIGMA_Z,
            "n_draws": 1000,
            "seed": 42,
        }
        async with async_client as c:
            first = await c.post("/qm/discrete-measurement", json=body)
            second = await c.post("/qm/discrete-measurement", json=body)
        assert first.status_code == 200
        counts = [o["count"] for o in first.json()["outcomes"]]
        assert sum(counts) == 1000
        assert first.json()["n_draws"] == 1000
        assert second.json() == first.json()

    async def test_no_counts_without_n_draws(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 1]},
                "operator": self.SIGMA_Z,
            })
        assert resp.json()["n_draws"] is None
        assert all(o["count"] is None for o in resp.json()["outcomes"])

    async def test_non_hermitian_operator(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0]},
                "operator": {"re": [[0, 1], [0, 0]]},
            })
        assert resp.status_code == 422

    async def test_dimension_mismatch(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0, 0]},
                "operator": self.SIGMA_Z,
            })
        assert resp.status_code == 422

    async def test_non_square_operator(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0]},
                "operator": {"re": [[1, 0, 0], [0, -1, 0]]},
            })
        assert resp.status_code == 422

    async def test_zero_state(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [0, 0]},
                "operator": self.SIGMA_Z,
            })
        assert resp.status_code == 422

    async def test_mismatched_re_and_im(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/discrete-measurement", json={
                "state": {"re": [1, 0], "im": [0]},
                "operator": self.SIGMA_Z,
            })
        assert resp.status_code == 422


class TestTrajectory:
    OMEGA = 1.0
    PERIOD = 2 * np.pi / OMEGA

    def body(self, **overrides):
        body = {
            "grid": {"x_min": -12, "x_max": 12, "n_points": 512},
            "potential": {"type": "harmonic", "params": {"omega": self.OMEGA}},
            "wavepacket": {"x0": 2.5, "k0": 0.0, "sigma": 1 / np.sqrt(2 * self.OMEGA)},
            "t_max": self.PERIOD,
            "dt": 0.005,
            "n_frames": 120,
        }
        body.update(overrides)
        return body

    async def test_returns_all_series_at_the_right_length(self, async_client):
        async with async_client as c:
            resp = await c.post(
                "/qm/trajectory",
                json=self.body(n_frames=60, include_classical=True),
            )
        assert resp.status_code == 200
        d = resp.json()

        for key in (
            "times",
            "mean_position",
            "mean_momentum",
            "spread_position",
            "spread_momentum",
            "uncertainty_product",
            "classical_position",
            "classical_momentum",
        ):
            assert len(d[key]) == 60, key

    async def test_mean_position_is_the_classical_cosine(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body())
        d = resp.json()

        t = np.array(d["times"])
        assert np.max(np.abs(np.array(d["mean_position"]) - 2.5 * np.cos(t))) < 1e-3

    async def test_classical_is_absent_unless_requested(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body())
        d = resp.json()
        assert d["classical_position"] is None
        assert d["classical_momentum"] is None

    async def test_ehrenfest_quantum_matches_classical(self, async_client):
        """With include_classical, the two series agree in a harmonic well."""
        async with async_client as c:
            resp = await c.post(
                "/qm/trajectory", json=self.body(include_classical=True)
            )
        d = resp.json()

        quantum = np.array(d["mean_position"])
        classical = np.array(d["classical_position"])
        assert np.max(np.abs(quantum - classical)) < 1e-3

    async def test_coherent_state_holds_its_width(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body())
        d = resp.json()

        sigma = 1 / np.sqrt(2 * self.OMEGA)
        assert np.allclose(d["spread_position"], sigma, atol=1e-3)
        assert np.allclose(d["uncertainty_product"], 0.5, atol=1e-3)

    async def test_response_carries_no_derived_extras(self, async_client):
        """The response is the series the request asked for, nothing bolted on."""
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body())
        assert "turning_points" not in resp.json()
        assert "coherent_width" not in resp.json()

    async def test_squeezed_state_breathes(self, async_client):
        async with async_client as c:
            resp = await c.post(
                "/qm/trajectory",
                json=self.body(wavepacket={"x0": 2.5, "k0": 0.0, "sigma": 0.4}),
            )
        d = resp.json()

        spread = np.array(d["spread_position"])
        assert spread.max() - spread.min() > 0.2
        # Still a valid state at every instant.
        assert np.all(np.array(d["uncertainty_product"]) >= 0.5 - 1e-6)

    async def test_energy(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body())
        assert resp.json()["energy"] == pytest.approx(3.625, rel=1e-3)

    async def test_classical_rejected_for_stepped_potentials(self, async_client):
        """A square wall has no finite-difference force; say so, don't return null."""
        async with async_client as c:
            resp = await c.post(
                "/qm/trajectory",
                json=self.body(
                    potential={"type": "infinite_well", "params": {"width": 6.0}},
                    wavepacket={"x0": 0.5, "k0": 1.0, "sigma": 0.5},
                    t_max=2.0,
                    include_classical=True,
                ),
            )
        assert resp.status_code == 422
        assert "include_classical" in str(resp.json())

    async def test_quantum_series_still_fine_for_stepped_potentials(self, async_client):
        async with async_client as c:
            resp = await c.post(
                "/qm/trajectory",
                json=self.body(
                    potential={"type": "infinite_well", "params": {"width": 6.0}},
                    wavepacket={"x0": 0.5, "k0": 1.0, "sigma": 0.5},
                    t_max=2.0,
                ),
            )
        assert resp.status_code == 200
        d = resp.json()
        assert len(d["mean_position"]) == 120
        assert np.all(np.isfinite(d["mean_position"]))

    async def test_rejects_bad_parameters(self, async_client):
        async with async_client as c:
            resp = await c.post("/qm/trajectory", json=self.body(t_max=-1.0))
        assert resp.status_code == 422


class TestEvolveViewWindow:
    """
    Solve wide, stream narrow. The grid must outrun the packet so the FFT
    never wraps it; the window is just what's worth drawing.
    """

    def request(self, **overrides):
        body = {
            "grid": {"x_min": -25, "x_max": 25, "n_points": 2048},
            "potential": {"type": "harmonic", "params": {"omega": 1.0}},
            "wavepacket": {"x0": 2.5, "k0": 0.0, "sigma": 0.7071067811865476},
            "t_max": 3.0,
            "dt": 0.005,
            "n_frames": 10,
        }
        body.update(overrides)
        return body

    def run(self, client, body):
        with client.websocket_connect("/qm/evolve") as ws:
            ws.send_text(json.dumps(body))
            metadata = json.loads(ws.receive_text())
            frames = []
            while True:
                msg = json.loads(ws.receive_text())
                if msg["type"] == "done":
                    break
                frames.append(msg)
        return metadata, frames

    def test_uncropped_streams_the_whole_grid(self, client):
        metadata, frames = self.run(client, self.request())
        assert len(metadata["x"]) == 2048
        assert len(frames[0]["probability_density"]) == 2048

    def test_window_crops_both_metadata_and_frames(self, client):
        metadata, frames = self.run(client, self.request(view_window=[-5.0, 5.0]))

        n = len(metadata["x"])
        assert n < 2048
        assert min(metadata["x"]) >= -5.0
        assert max(metadata["x"]) <= 5.0
        assert len(metadata["potential"]) == n
        for f in frames:
            assert len(f["probability_density"]) == n

    def test_full_grid_bounds_are_still_reported(self, client):
        metadata, _ = self.run(client, self.request(view_window=[-5.0, 5.0]))
        assert metadata["grid_bounds"] == [-25.0, 25.0]

    def test_norm_stays_a_full_grid_integral(self, client):
        """Cropping the view must not make the packet look like it lost norm."""
        _, frames = self.run(client, self.request(view_window=[-3.0, 3.0]))
        for f in frames:
            assert f["norm"] == pytest.approx(1.0, abs=1e-3)

    def test_cropping_does_not_change_the_physics(self, client):
        """The solve runs on the full grid either way."""
        _, wide = self.run(client, self.request())
        metadata, cropped = self.run(client, self.request(view_window=[-5.0, 5.0]))

        x_full = np.array(json.loads(json.dumps(wide[0]["probability_density"])))
        # Locate the cropped window inside the full grid and compare pointwise.
        n = len(metadata["x"])
        start = int(np.argmin(np.abs(np.linspace(-25, 25, 2048) - metadata["x"][0])))
        assert np.allclose(
            x_full[start : start + n], cropped[0]["probability_density"], atol=1e-12
        )

    def test_rejects_inverted_window(self, client):
        with pytest.raises(Exception):
            self.run(client, self.request(view_window=[5.0, -5.0]))

    def test_rejects_window_off_the_grid(self, client):
        with pytest.raises(Exception):
            self.run(client, self.request(view_window=[100.0, 200.0]))


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
            # inequality) -- see eigora.qm's scattering tests.
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
