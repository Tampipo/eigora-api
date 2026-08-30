# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the statistical physics router.

Each endpoint is checked against something the library is not asked for: a
closed form, an exact solve, or a limit. The Monte Carlo endpoint is run with
deliberately small statistics, so its assertions are loose on purpose -- what
is being tested is that the plumbing carries the right physics, not that a
short chain converges.
"""

import json
import math

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from eigora_api.main import app
from eigora_api.routers.statphys import _sweep_point

# Peak of the two-level heat capacity: x = splitting/T solving e^x = (x+2)/(x-2).
SCHOTTKY_X = 2.399357
SCHOTTKY_C = 0.439229


@pytest.fixture
def async_client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def client():
    return TestClient(app)


class TestThermodynamics:
    async def test_two_level_reproduces_the_schottky_peak(self, async_client):
        peak = 1.0 / SCHOTTKY_X
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "two_level", "splitting": 1.0},
                "temperatures": {
                    "minimum": peak, "maximum": peak * 1.0001, "points": 2
                },
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["heat_capacity"][0] == pytest.approx(SCHOTTKY_C, abs=1e-4)
        assert data["potential_name"] == "free_energy"
        assert data["states"] == 2

    async def test_entropy_saturates_at_log_two(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "two_level", "splitting": 1.0},
                "temperatures": {"minimum": 1e5, "maximum": 1e6, "points": 2},
            })
        assert resp.json()["entropy"][-1] == pytest.approx(math.log(2), abs=1e-4)

    async def test_copies_are_extensive(self, async_client):
        async def sweep(copies):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post("/v1/statphys/thermodynamics", json={
                    "system": {"type": "two_level", "copies": copies},
                    "temperatures": {"minimum": 0.7, "maximum": 0.7001, "points": 2},
                })
            return resp.json()

        one, many = await sweep(1), await sweep(20)
        assert many["entropy"][0] == pytest.approx(20 * one["entropy"][0], rel=1e-9)
        assert many["energy"][0] == pytest.approx(20 * one["energy"][0], rel=1e-9)

    async def test_a_magnetic_ensemble_reports_the_magnetisation(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "spin", "spin": 0.5},
                "ensemble": {"type": "magnetic", "magnetic_field": 0.5},
                "temperatures": {"minimum": 1.0, "maximum": 1.0001, "points": 2},
            })
        data = resp.json()
        assert data["magnetisation"][0] == pytest.approx(0.5 * math.tanh(0.25))
        assert data["susceptibility"][0] > 0.0
        assert data["potential_name"] == "magnetic_free_energy"

    async def test_a_system_without_the_freed_variable_is_refused(self, async_client):
        """
        A two-level system carries no magnetisation, so a magnetic ensemble
        cannot be applied to it. That is a 422, not a silent zero.
        """
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "two_level"},
                "ensemble": {"type": "magnetic", "magnetic_field": 0.5},
            })
        assert resp.status_code == 422
        assert "magnetisation" in resp.json()["detail"]

    async def test_an_unbounded_spectrum_reports_no_state_count(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "harmonic", "omega": 1.0},
                "temperatures": {"minimum": 0.5, "maximum": 2.0, "points": 5},
            })
        data = resp.json()
        assert data["states"] is None
        # <E> -> omega/2 as T -> 0, and grows from there.
        assert data["energy"][0] > 0.5
        assert data["energy"][-1] > data["energy"][0]

    async def test_temperature_bounds_are_validated(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "temperatures": {"minimum": 5.0, "maximum": 1.0},
            })
        assert resp.status_code == 422


class TestCoexistence:
    async def test_the_pressure_matches_the_closed_form(self, async_client):
        """
        P = (k_B T / L^3)(1 - e^{-b hw})^3 exp(-b(eps_c - 3 hw/2)), worked out
        here from the request parameters and nothing else.
        """
        omega, cohesion, mass_amu = 90.0, 1066.0, 39.948
        async with async_client as c:
            resp = await c.post("/v1/statphys/coexistence", json={
                "omega": omega, "cohesion": cohesion, "mass_amu": mass_amu,
                "temperatures": {
                    "minimum": 60.0, "maximum": 60.0001, "points": 2,
                    "logarithmic": False,
                },
            })
        data = resp.json()
        hbar, boltzmann, amu = 1.054571817e-34, 1.380649e-23, 1.66053906660e-27
        temperature, beta = 60.0, 1.0 / 60.0
        wavelength = hbar * math.sqrt(
            2.0 * math.pi / (mass_amu * amu * boltzmann * temperature)
        )
        expected = (
            boltzmann * temperature / wavelength**3
            * (1.0 - math.exp(-beta * omega)) ** 3
            * math.exp(-beta * (cohesion - 1.5 * omega))
        )
        assert data["pressure"][0] == pytest.approx(expected, rel=1e-9)
        assert data["latent_heat"] == pytest.approx(cohesion - 1.5 * omega)

    async def test_the_curve_rises_with_temperature(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/coexistence", json={
                "temperatures": {
                    "minimum": 45.0, "maximum": 80.0, "points": 12,
                    "logarithmic": False,
                },
            })
        pressure = resp.json()["pressure"]
        assert all(b > a for a, b in zip(pressure, pressure[1:]))

    async def test_log_p_against_one_over_t_is_nearly_straight(self, async_client):
        """Clausius-Clapeyron: the slope is the latent heat."""
        async with async_client as c:
            resp = await c.post("/v1/statphys/coexistence", json={
                "temperatures": {
                    "minimum": 45.0, "maximum": 80.0, "points": 40,
                    "logarithmic": False,
                },
            })
        data = resp.json()
        inverse = [1.0 / t for t in data["temperatures"]]
        logs = [math.log(p) for p in data["pressure"]]
        slope = (logs[-1] - logs[0]) / (inverse[-1] - inverse[0])
        assert -slope == pytest.approx(data["latent_heat"], rel=0.15)


class TestGas:
    async def test_fermions_fit_their_own_chemical_potential(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas", json={
                "box": 5.0, "particles": 30, "spin": 0.5,
                "temperature": 1500.0, "samples": 400, "burn_in_sweeps": 100,
                "seed": 7,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"] == "fermi"
        assert all(0.0 <= n <= 1.0001 for n in data["occupation"])
        assert data["chemical_potential_fitted"] == pytest.approx(
            data["chemical_potential_exact"], rel=0.15
        )
        assert 0.0 < data["acceptance"] <= 1.0

    async def test_bosons_may_exceed_one_per_orbital(self, async_client):
        """The occupation ceiling is the whole difference, so check it is gone."""
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas", json={
                "box": 5.0, "particles": 30, "spin": 1.0,
                "temperature": 400.0, "samples": 400, "burn_in_sweeps": 100,
                "seed": 8,
            })
        data = resp.json()
        assert data["statistics"] == "bose"
        assert max(data["occupation"]) > 1.0

    async def test_the_spin_sets_the_orbital_count(self, async_client):
        """`2s + 1` states per energy, so spin 1 has half again as many as 1/2."""
        async def orbitals(spin):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post("/v1/statphys/gas", json={
                    "box": 4.0, "particles": 8, "spin": spin,
                    "temperature": 2000.0, "samples": 120, "burn_in_sweeps": 20,
                    "seed": 3,
                })
            return resp.json()["orbitals"]

        assert await orbitals(1.0) == pytest.approx(1.5 * await orbitals(0.5), rel=0.2)

    async def test_a_quarter_spin_is_refused(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas", json={"spin": 0.25})
        assert resp.status_code == 422


class TestIsingSocket:
    def test_it_streams_metadata_then_frames_then_done(self, client):
        with client.websocket_connect("/v1/statphys/ising") as socket:
            socket.send_text(json.dumps({
                "size": 8, "temperature": 2.5, "frames": 3,
                "sweeps_per_frame": 1, "seed": 5,
            }))
            metadata = socket.receive_json()
            assert metadata["type"] == "metadata"
            assert metadata["sites"] == 64
            assert metadata["critical_temperature"] == pytest.approx(
                2.0 / math.log(1.0 + math.sqrt(2.0))
            )

            for index in range(3):
                frame = socket.receive_json()
                assert frame["type"] == "frame"
                assert frame["frame"] == index
                assert len(frame["spins"]) == 8
                assert all(len(row) == 8 for row in frame["spins"])
                assert all(s in (-1, 1) for row in frame["spins"] for s in row)
                assert -2.0 <= frame["energy_per_site"] <= 2.0
                assert -1.0 <= frame["magnetisation_per_site"] <= 1.0

            assert socket.receive_json()["type"] == "done"

    def test_a_cold_lattice_ends_up_ordered(self, client):
        """Well below T_c a small lattice should align, so |m| is near one."""
        with client.websocket_connect("/v1/statphys/ising") as socket:
            socket.send_text(json.dumps({
                "size": 8, "temperature": 0.6, "frames": 40,
                "sweeps_per_frame": 5, "seed": 11,
            }))
            socket.receive_json()
            last = None
            for _ in range(40):
                last = socket.receive_json()
            assert abs(last["magnetisation_per_site"]) > 0.85

    def test_a_bad_request_reports_an_error_frame(self, client):
        with client.websocket_connect("/v1/statphys/ising") as socket:
            socket.send_text(json.dumps({"size": 2}))
            message = socket.receive_json()
            assert message["type"] == "error"


class TestGasSocket:
    def test_it_streams_metadata_then_frames_then_done(self, client):
        with client.websocket_connect("/v1/statphys/gas") as socket:
            socket.send_text(json.dumps({
                "box": 4.0, "particles": 12, "spin": 0.5,
                "temperature": 1500.0, "samples": 120, "burn_in_sweeps": 20,
                "frames": 4, "seed": 5,
            }))
            metadata = socket.receive_json()
            assert metadata["type"] == "metadata"
            assert metadata["statistics"] == "fermi"
            assert metadata["frames"] == 4
            assert len(metadata["energies"]) == len(metadata["degeneracies"])
            assert len(metadata["distribution"]) == len(metadata["energies"])
            assert sum(metadata["degeneracies"]) == metadata["orbitals"]

            for index in range(4):
                frame = socket.receive_json()
                assert frame["type"] == "frame"
                assert frame["frame"] == index
                assert len(frame["occupation"]) == len(metadata["energies"])
                assert all(0.0 <= n <= 1.0001 for n in frame["occupation"])
                assert 0.0 < frame["acceptance"] <= 1.0

            assert socket.receive_json()["type"] == "done"

    def test_the_error_bars_close_as_the_chain_runs(self, client):
        """
        The running average is the point of streaming: the same statistic,
        measured over more samples, must get tighter -- and tighter by 1/sqrt(n),
        which is the claim the error bars actually make.

        The burn-in has to be generous for this to hold. On a chain that has not
        decorrelated the sample spread is itself an underestimate, so the bars
        can widen for a few frames before they start closing.
        """
        with client.websocket_connect("/v1/statphys/gas") as socket:
            socket.send_text(json.dumps({
                "box": 4.0, "particles": 12, "spin": 0.5,
                "temperature": 1500.0, "samples": 800, "burn_in_sweeps": 200,
                "frames": 8, "seed": 13,
            }))
            socket.receive_json()
            frames = [socket.receive_json() for _ in range(8)]

        assert [f["samples"] for f in frames] == sorted(f["samples"] for f in frames)
        bars = [max(f["occupation_error"]) for f in frames]
        assert bars == sorted(bars, reverse=True)
        assert bars[0] / bars[-1] == pytest.approx(math.sqrt(8.0), rel=0.3)

    def test_the_running_fit_lands_near_the_exact_chemical_potential(self, client):
        with client.websocket_connect("/v1/statphys/gas") as socket:
            socket.send_text(json.dumps({
                "box": 5.0, "particles": 30, "spin": 0.5,
                "temperature": 1500.0, "samples": 400, "burn_in_sweeps": 100,
                "frames": 5, "seed": 7,
            }))
            metadata = socket.receive_json()
            last = [socket.receive_json() for _ in range(5)][-1]

        assert last["chemical_potential_fitted"] == pytest.approx(
            metadata["chemical_potential_exact"], rel=0.15
        )

    def test_a_bad_request_reports_an_error_frame(self, client):
        with client.websocket_connect("/v1/statphys/gas") as socket:
            socket.send_text(json.dumps({"spin": 0.25}))
            message = socket.receive_json()
            assert message["type"] == "error"


class TestGasSweep:
    async def test_a_temperature_sweep_follows_the_closed_forms(self, async_client):
        """
        Below the Fermi temperature mu tracks Sommerfeld and falls with T;
        the simulation is never told either fact.
        """
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas-sweep", json={
                "box": 4.0, "spin": 0.5, "vary": "temperature",
                "values": [800.0, 1600.0, 3200.0], "particles": 20,
                "samples": 200, "burn_in_sweeps": 60, "seed": 4,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["vary"] == "temperature"
        assert data["statistics"] == "fermi"
        assert [p["temperature"] for p in data["points"]] == [800.0, 1600.0, 3200.0]
        assert all(p["particles"] == 20 for p in data["points"])

        chemical = [p["chemical_potential_exact"] for p in data["points"]]
        assert chemical == sorted(chemical, reverse=True)
        for point in data["points"]:
            assert point["sommerfeld"] is not None
            assert point["maxwell_boltzmann"] is not None
            assert point["chemical_potential_fitted"] == pytest.approx(
                point["chemical_potential_exact"], rel=0.2
            )

        cold = data["points"][0]
        assert cold["chemical_potential_exact"] == pytest.approx(
            cold["sommerfeld"], rel=0.15
        )

    async def test_a_particle_sweep_raises_the_fermi_temperature(self, async_client):
        """More fermions in the same box means a deeper sea, so Tf climbs."""
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas-sweep", json={
                "box": 4.0, "spin": 0.5, "vary": "particles",
                "values": [10, 20, 40], "temperature": 1500.0,
                "samples": 150, "burn_in_sweeps": 40, "seed": 6,
            })
        data = resp.json()
        assert data["vary"] == "particles"
        assert [p["particles"] for p in data["points"]] == [10, 20, 40]
        assert all(p["temperature"] == 1500.0 for p in data["points"])

        fermi = [p["fermi_temperature"] for p in data["points"]]
        assert fermi == sorted(fermi)

    async def test_bosons_get_no_fermion_closed_forms(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas-sweep", json={
                "box": 4.0, "spin": 1.0, "vary": "temperature",
                "values": [600.0], "particles": 12,
                "samples": 150, "burn_in_sweeps": 40, "seed": 9,
            })
        data = resp.json()
        assert data["statistics"] == "bose"
        assert data["points"][0]["sommerfeld"] is None
        assert data["points"][0]["maxwell_boltzmann"] is None

    def test_one_point_runs_standalone(self):
        """
        The worker is called directly here, not through the pool. The pool path
        is covered by the sweeps above; this covers the physics inside a worker
        without a subprocess between the assertion and the code.
        """
        point = _sweep_point((4.0, 0.5, 20, 1200.0, 200, 60, 4))
        assert point["statistics"] == "fermi"
        assert point["particles"] == 20
        assert point["temperature"] == 1200.0
        assert point["chemical_potential_fitted"] == pytest.approx(
            point["chemical_potential_exact"], rel=0.2
        )
        assert point["sommerfeld"] == pytest.approx(
            point["fermi_temperature"] * (
                1.0 - (math.pi**2 / 12.0)
                * (1200.0 / point["fermi_temperature"]) ** 2
            )
        )

    async def test_fractional_particle_counts_are_refused(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/gas-sweep", json={
                "vary": "particles", "values": [10.5],
            })
        assert resp.status_code == 422


class TestOtherSystems:
    """
    The system and ensemble types the endpoints above never reach.

    Each is checked against a limit rather than a stored number: a rotor and an
    ideal gas both give one unit of energy per quadratic degree of freedom at
    high temperature, and the isobaric ensemble must report the volume it was
    given a pressure for.
    """

    async def test_a_rotor_approaches_one_unit_of_heat_capacity(self, async_client):
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {"type": "rotor", "rotational_constant": 1.0},
                "temperatures": {"minimum": 40.0, "maximum": 60.0, "points": 3},
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["heat_capacity"][-1] == pytest.approx(1.0, rel=0.05)
        assert data["energy"][-1] == pytest.approx(
            data["temperatures"][-1], rel=0.05
        )

    async def test_an_ideal_gas_carries_three_halves_nt(self, async_client):
        """Equipartition: three quadratic degrees of freedom per particle."""
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {
                    "type": "ideal_gas", "particles": 10, "volume": 100.0,
                },
                "temperatures": {"minimum": 1.0, "maximum": 4.0, "points": 4},
            })
        data = resp.json()
        for temperature, energy, heat in zip(
            data["temperatures"], data["energy"], data["heat_capacity"]
        ):
            assert energy == pytest.approx(1.5 * 10 * temperature, rel=1e-6)
            assert heat == pytest.approx(1.5 * 10, rel=1e-6)

    async def test_the_isobaric_ensemble_is_refused_by_every_system(self, async_client):
        """
        No system this endpoint can build reports a volume per microstate --
        `IdealGas` carries volume as a parameter, not as a free variable -- so
        the isobaric ensemble is a 422 rather than a plausible wrong number.
        """
        async with async_client as c:
            resp = await c.post("/v1/statphys/thermodynamics", json={
                "system": {
                    "type": "ideal_gas", "particles": 10, "volume": 100.0,
                },
                "ensemble": {"type": "isobaric", "pressure": 0.1},
                "temperatures": {"minimum": 1.0, "maximum": 2.0, "points": 2},
            })
        assert resp.status_code == 422
        assert "volume" in resp.json()["detail"]
