# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Statistical physics router.

Mounted at /v1 (see main.py), so paths below are relative to that prefix.

POST /statphys/thermodynamics  — U, C, S and the potential over a temperature sweep
POST /statphys/coexistence     — solid-gas equilibrium and the sublimation curve
POST /statphys/gas             — Monte Carlo occupations against the closed form
WS   /statphys/ising           — stream an Ising lattice as it orders

Units are the library's: k_B = hbar = 1, so a temperature is an energy.
`coexistence` is the exception -- it needs a real mass for the thermal
wavelength, so it quotes energies in kelvin and pressure in pascals.
"""

import asyncio
import json
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from scipy.optimize import curve_fit

from eigora.statphys import (
    Canonical,
    Generalised,
    HarmonicMode,
    IdealGas,
    IsothermalIsobaric,
    Field as ConjugateField,
    Magnetic,
    NLevel,
    Rotor,
    Spin,
    System,
    TwoLevel,
    equilibrium,
)
from eigora.statphys.monte_carlo import Gas, IsingLattice, metropolis

from eigora_api.schemas.statphys import (
    CoexistenceRequest,
    CoexistenceResponse,
    EnsembleSchema,
    EnsembleType,
    GasMetadata,
    GasRequest,
    GasResponse,
    GasStreamFrame,
    GasStreamRequest,
    GasSweepPoint,
    GasSweepRequest,
    GasSweepResponse,
    IsingFrame,
    IsingMetadata,
    IsingRequest,
    SystemSchema,
    SystemType,
    TemperatureRange,
    ThermodynamicsRequest,
    ThermodynamicsResponse,
)

router = APIRouter(prefix="/statphys", tags=["Statistical Physics"])

HBAR = 1.054571817e-34
BOLTZMANN = 1.380649e-23
ATOMIC_MASS = 1.66053906660e-27
ELECTRON_MASS = 9.1093837015e-31
CRITICAL = 2.0 / math.log(1.0 + math.sqrt(2.0))


def _temperatures(spec: TemperatureRange) -> np.ndarray:
    if spec.logarithmic:
        return np.logspace(
            math.log10(spec.minimum), math.log10(spec.maximum), spec.points
        )
    return np.linspace(spec.minimum, spec.maximum, spec.points)


def _build_system(spec: SystemSchema) -> System:
    """One subsystem, raised to `copies` distinguishable tensor factors."""
    if spec.type is SystemType.two_level:
        block: System = TwoLevel(splitting=spec.splitting)
    elif spec.type is SystemType.harmonic:
        block = HarmonicMode(omega=spec.omega)
    elif spec.type is SystemType.spin:
        block = Spin(spec.spin)
    elif spec.type is SystemType.rotor:
        block = Rotor(spec.rotational_constant)
    else:
        block = IdealGas(particles=spec.particles, volume=spec.volume)
    return block if spec.copies == 1 else block**spec.copies


def _build_ensemble(spec: EnsembleSchema, temperature: float):
    if spec.type is EnsembleType.canonical:
        return Canonical(temperature)
    if spec.type is EnsembleType.magnetic:
        return Magnetic(temperature, spec.magnetic_field)
    return IsothermalIsobaric(temperature, spec.pressure)


@router.post("/thermodynamics", response_model=ThermodynamicsResponse)
def thermodynamics(req: ThermodynamicsRequest) -> ThermodynamicsResponse:
    """
    Sweep a system in an ensemble over temperature.

    The ensemble decides what is reported: a magnetic one adds the
    magnetisation and its susceptibility, an isobaric one the mean volume.
    Asking for a system the ensemble cannot be applied to -- a two-level
    system in a magnetic field, whose levels carry no magnetisation -- is a
    422 rather than a wrong number.
    """
    system = _build_system(req.system)
    grid = _temperatures(req.temperatures)

    try:
        states = [equilibrium(system, _build_ensemble(req.ensemble, float(t)))
                  for t in grid]
        energy = [s.energy for s in states]
        heat = [s.heat_capacity for s in states]
        entropy = [s.entropy for s in states]
        potential = [s.potential for s in states]
        name = states[0].potential_name

        magnetisation = susceptibility = volume = None
        if req.ensemble.type is EnsembleType.magnetic:
            magnetisation = [s.magnetisation for s in states]
            susceptibility = [s.susceptibility for s in states]
        elif req.ensemble.type is EnsembleType.isobaric:
            volume = [s.mean("volume") for s in states]
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error))

    return ThermodynamicsResponse(
        temperatures=grid.tolist(),
        energy=energy,
        heat_capacity=heat,
        entropy=entropy,
        potential=potential,
        potential_name=name,
        magnetisation=magnetisation,
        susceptibility=susceptibility,
        volume=volume,
        states=getattr(system, "n_states", None),
    )


@router.post("/coexistence", response_model=CoexistenceResponse)
def coexistence(req: CoexistenceRequest) -> CoexistenceResponse:
    """
    Solid-gas equilibrium: `mu_crystal = mu_gas` gives the sublimation curve.

    The crystal is `HarmonicMode(omega) ** 3` for the three modes of one atom,
    times a single level at `-eps_c` for its binding -- log Z adds over blocks,
    so that shift is exactly the `+beta eps_c` the energy carries. The gas is
    an ideal gas, and equating the two chemical potentials closes it:

        P = (k_B T / Lambda^3) (1 - e^{-beta hbar omega})^3
            exp(-beta (eps_c - 3 hbar omega / 2))
    """
    atom = HarmonicMode(omega=req.omega) ** 3 * NLevel([-req.cohesion])
    mass = req.mass_amu * ATOMIC_MASS
    grid = _temperatures(req.temperatures)

    chemical, pressures = [], []
    for temperature in grid:
        beta = 1.0 / float(temperature)
        # mu_c = F(N) - F(N-1) = -T log zeta_c, and independent of N.
        mu = -float(temperature) * atom.log_z(beta)
        wavelength = HBAR * math.sqrt(
            2.0 * math.pi / (mass * BOLTZMANN * float(temperature))
        )
        density = math.exp(mu / float(temperature)) / wavelength**3
        chemical.append(mu)
        pressures.append(density * BOLTZMANN * float(temperature))

    return CoexistenceResponse(
        temperatures=grid.tolist(),
        chemical_potential=chemical,
        pressure=pressures,
        latent_heat=req.cohesion - 1.5 * req.omega,
    )


def _box_orbitals(box_nm: float, particles: int, temperature: float, spin: float):
    """
    Every `(nx, ny, nz)` inside a thermal cutoff, with the Fermi sea guaranteed.

    The cutoff keeps states up to `Ef + 6T`, but with a floor at the Fermi
    radius plus two shells: at low temperature the thermal criterion alone
    would barely hold the particles, and the gas would be squeezed by the
    truncation rather than by the physics.
    """
    length = box_nm * 1e-9
    multiplicity = round(2.0 * spin + 1.0)
    scale = (HBAR**2 / (2.0 * ELECTRON_MASS * BOLTZMANN)) * (
        2.0 * math.pi / length
    ) ** 2
    fermi = (HBAR**2 / (2.0 * ELECTRON_MASS * BOLTZMANN)) * (
        3.0 * math.pi**2 * particles / length**3
    ) ** (2.0 / 3.0)
    ceiling = fermi + 6.0 * temperature
    limit = max(
        int(math.sqrt(ceiling / scale)),
        int(math.ceil(math.sqrt(fermi / scale))) + 2,
    )
    axis = np.arange(-limit, limit + 1)
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    spatial = scale * sum(component**2 for component in grid).ravel()
    return np.sort(spatial), multiplicity, fermi


def _prepare(box, particles, spin, temperature, seed):
    """
    Everything both the request and the stream need before sampling starts.

    Returns the sampler, the distinct orbital energies with their
    degeneracies, the Fermi temperature, and the exactly-solved chemical
    potential -- the reference the fit is later checked against.
    """
    spatial, _, fermi = _box_orbitals(box, particles, temperature, spin)
    rng = np.random.default_rng(seed)
    sampler = Gas.of_spin(spatial, particles, spin, rng=rng)
    levels, inverse, counts = np.unique(
        np.round(sampler.energies, 6), return_inverse=True, return_counts=True
    )
    sign = sampler.statistics.grand_sign
    exact = _solve_chemical_potential(levels, counts, temperature, particles, sign)
    return sampler, rng, levels, inverse, counts, fermi, exact


def _distribution(energy, chemical_potential, temperature, sign):
    return 1.0 / (np.exp((energy - chemical_potential) / temperature) + sign)


def _pool(mean, spread, inverse, counts):
    """Average over orbitals of equal energy, as the distribution is a function
    of energy alone."""
    pooled = np.bincount(inverse, weights=mean) / counts
    error = np.sqrt(np.bincount(inverse, weights=spread**2)) / counts
    return pooled, error


def _fit(levels, pooled, error, guess, temperature, sign):
    """Least squares for `mu`, or `None` when the chain has nothing stable yet."""
    try:
        fitted, covariance = curve_fit(
            lambda e, mu: _distribution(e, mu, temperature, sign),
            levels, pooled, p0=[guess],
            sigma=np.maximum(error, 1e-4), absolute_sigma=True,
        )
        return float(fitted[0]), float(math.sqrt(covariance[0, 0]))
    except (RuntimeError, ValueError):
        return None, None


@router.post("/gas", response_model=GasResponse)
def gas(req: GasRequest) -> GasResponse:
    """
    Sample a gas of spin-`s` particles and fit its chemical potential.

    Occupations are measured by Metropolis, averaged over orbitals of equal
    energy, and fitted to the distribution with `mu` free. The exact `mu` is
    solved separately and returned alongside, so the fit is checked rather
    than trusted.
    """
    try:
        sampler, rng, levels, inverse, counts, fermi, exact = _prepare(
            req.box, req.particles, req.spin, req.temperature, req.seed
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    ensemble = Canonical(req.temperature)
    metropolis(sampler, ensemble, req.burn_in_sweeps * req.particles, rng=rng)

    total = np.zeros(sampler.energies.size)
    squares = np.zeros(sampler.energies.size)
    accepted = attempted = 0
    for _ in range(req.samples):
        run = metropolis(sampler, ensemble, req.particles, rng=rng)
        accepted += run.accepted
        attempted += run.steps - run.blocked
        occupation = sampler.occupations()
        total += occupation
        squares += occupation**2

    mean = total / req.samples
    spread = np.sqrt(
        np.maximum(squares / req.samples - mean**2, 0.0) / req.samples
    )
    pooled, error = _pool(mean, spread, inverse, counts)
    sign = sampler.statistics.grand_sign
    fitted, uncertainty = _fit(levels, pooled, error, exact, req.temperature, sign)

    return GasResponse(
        energies=levels.tolist(),
        occupation=pooled.tolist(),
        occupation_error=error.tolist(),
        distribution=_distribution(levels, exact, req.temperature, sign).tolist(),
        statistics=sampler.statistics.name,
        fermi_temperature=fermi,
        chemical_potential_fitted=fitted if fitted is not None else exact,
        chemical_potential_error=uncertainty if uncertainty is not None else 0.0,
        chemical_potential_exact=exact,
        orbitals=int(sampler.energies.size),
        acceptance=accepted / attempted if attempted else 0.0,
    )


def _sweep_point(argument):
    """
    One converged point of a sweep, run in its own process.

    Module level and taking a plain tuple, because that is what has to survive
    pickling to reach a worker.
    """
    box, spin, particles, temperature, samples, burn_in, seed = argument
    sampler, rng, levels, inverse, counts, fermi, exact = _prepare(
        box, particles, spin, temperature, seed
    )
    ensemble = Canonical(temperature)
    metropolis(sampler, ensemble, burn_in * particles, rng=rng)

    total = np.zeros(sampler.energies.size)
    squares = np.zeros(sampler.energies.size)
    for _ in range(samples):
        metropolis(sampler, ensemble, particles, rng=rng)
        occupation = sampler.occupations()
        total += occupation
        squares += occupation**2
    mean = total / samples
    spread = np.sqrt(np.maximum(squares / samples - mean**2, 0.0) / samples)
    pooled, error = _pool(mean, spread, inverse, counts)
    sign = sampler.statistics.grand_sign
    fitted, uncertainty = _fit(levels, pooled, error, exact, temperature, sign)

    sommerfeld = boltzmann = None
    if sign > 0:                       # the 3D fermion limits, and only those
        sommerfeld = fermi * (1.0 - (math.pi**2 / 12.0) * (temperature / fermi) ** 2)
        boltzmann = 1.5 * temperature * math.log(
            4.0 * math.pi * fermi
            / ((6.0 * math.pi**2) ** (2.0 / 3.0) * temperature)
        )
    return {
        "temperature": temperature,
        "particles": particles,
        "fermi_temperature": fermi,
        "chemical_potential_fitted": fitted if fitted is not None else exact,
        "chemical_potential_error": uncertainty if uncertainty is not None else 0.0,
        "chemical_potential_exact": exact,
        "sommerfeld": sommerfeld,
        "maxwell_boltzmann": boltzmann,
        "statistics": sampler.statistics.name,
    }


@router.post("/gas-sweep", response_model=GasSweepResponse)
def gas_sweep(req: GasSweepRequest) -> GasSweepResponse:
    """
    Converged chemical potential at every value of the swept parameter.

    The points are independent, so they run in parallel processes -- one
    request instead of a round trip per point, and wall time set by the
    slowest point rather than their sum. Against them come the closed forms
    the simulation never sees: Sommerfeld below the Fermi temperature and
    Maxwell-Boltzmann above it, both for fermions only.
    """
    jobs = []
    for index, value in enumerate(req.values):
        temperature = value if req.vary == "temperature" else req.temperature
        particles = req.particles if req.vary == "temperature" else int(value)
        seed = None if req.seed is None else req.seed + index
        jobs.append((
            req.box, req.spin, particles, temperature,
            req.samples, req.burn_in_sweeps, seed,
        ))

    try:
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(_sweep_point, jobs))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return GasSweepResponse(
        vary=req.vary,
        statistics=results[0]["statistics"],
        points=[GasSweepPoint(**{
            key: value for key, value in point.items() if key != "statistics"
        }) for point in results],
    )


def _solve_chemical_potential(levels, counts, temperature, particles, sign) -> float:
    """
    The `mu` putting `<N>` at the requested particle number, by bisection.

    Bosons must stay strictly below the lowest orbital, where the occupation
    diverges, so the bracket is capped there rather than grown outward.
    """
    ceiling = (
        float(levels[0]) - 1e-9
        if sign < 0
        else float(levels[-1]) + 50.0 * temperature
    )

    def occupied(mu: float) -> float:
        return float(np.sum(counts / (np.exp((levels - mu) / temperature) + sign)))

    low = float(levels[0]) - 50.0 * temperature
    high = ceiling
    for _ in range(200):
        middle = 0.5 * (low + high)
        if occupied(middle) < particles:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


@router.websocket("/gas")
async def gas_stream(websocket: WebSocket) -> None:
    """
    Stream one chain as its distribution forms.

    Protocol mirrors /ising: metadata once, then a frame per batch of samples,
    then done. Each frame carries the running average over every sample so
    far, so the error bars visibly close -- which is the thing a converged
    POST response cannot show.
    """
    await websocket.accept()
    try:
        req = GasStreamRequest.model_validate_json(await websocket.receive_text())
        sampler, rng, levels, inverse, counts, fermi, exact = _prepare(
            req.box, req.particles, req.spin, req.temperature, req.seed
        )
        sign = sampler.statistics.grand_sign
        ensemble = Canonical(req.temperature)

        await websocket.send_text(json.dumps({
            "type": "metadata",
            **GasMetadata(
                energies=levels.tolist(),
                degeneracies=counts.tolist(),
                statistics=sampler.statistics.name,
                fermi_temperature=fermi,
                chemical_potential_exact=exact,
                distribution=_distribution(
                    levels, exact, req.temperature, sign
                ).tolist(),
                orbitals=int(sampler.energies.size),
                particles=req.particles,
                temperature=req.temperature,
                frames=req.frames,
            ).model_dump(),
        }))

        metropolis(sampler, ensemble, req.burn_in_sweeps * req.particles, rng=rng)
        total = np.zeros(sampler.energies.size)
        squares = np.zeros(sampler.energies.size)
        seen = accepted = attempted = 0
        per_frame = max(1, req.samples // req.frames)

        for index in range(req.frames):
            for _ in range(per_frame):
                run = metropolis(sampler, ensemble, req.particles, rng=rng)
                accepted += run.accepted
                attempted += run.steps - run.blocked
                occupation = sampler.occupations()
                total += occupation
                squares += occupation**2
                seen += 1
            mean = total / seen
            spread = np.sqrt(np.maximum(squares / seen - mean**2, 0.0) / seen)
            pooled, error = _pool(mean, spread, inverse, counts)
            fitted, uncertainty = _fit(
                levels, pooled, error, exact, req.temperature, sign
            )
            await websocket.send_text(json.dumps({
                "type": "frame",
                **GasStreamFrame(
                    frame=index,
                    samples=seen,
                    occupation=pooled.tolist(),
                    occupation_error=error.tolist(),
                    chemical_potential_fitted=fitted,
                    chemical_potential_error=uncertainty,
                    acceptance=accepted / attempted if attempted else 0.0,
                ).model_dump(),
            }))
            await asyncio.sleep(0)          # let the event loop breathe

        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        return
    except Exception as error:
        await websocket.send_text(json.dumps({"type": "error", "detail": str(error)}))
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@router.websocket("/ising")
async def ising(websocket: WebSocket) -> None:
    """
    Stream a 2D Ising lattice as it orders.

    Protocol:
    1. Client connects and sends IsingRequest as JSON
    2. Server sends { "type": "metadata", ... } once
    3. Server streams { "type": "frame", ... } per batch of sweeps
    4. Server sends { "type": "done" }

    The client may send { "temperature": T } at any point to change it mid-run,
    which is the whole reason this is a socket: quenching a lattice through
    T_c and watching domains coarsen is the demonstration.
    """
    await websocket.accept()
    try:
        req = IsingRequest.model_validate_json(await websocket.receive_text())
        rng = np.random.default_rng(req.seed)
        lattice = IsingLattice((req.size, req.size), req.coupling, rng=rng)
        steps = req.sweeps_per_frame * lattice.sites

        await websocket.send_text(json.dumps({
            "type": "metadata",
            **IsingMetadata(
                size=req.size,
                critical_temperature=CRITICAL,
                sites=lattice.sites,
                frames=req.frames,
                sweeps_per_frame=req.sweeps_per_frame,
            ).model_dump(),
        }))

        temperature = req.temperature
        for index in range(req.frames):
            try:
                message = json.loads(
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.0)
                )
                temperature = float(message.get("temperature", temperature))
            except (asyncio.TimeoutError, ValueError, KeyError):
                pass

            run = metropolis(lattice, Canonical(temperature), steps, rng=rng)
            await websocket.send_text(json.dumps({
                "type": "frame",
                **IsingFrame(
                    frame=index,
                    sweep=index * req.sweeps_per_frame,
                    temperature=temperature,
                    spins=lattice.spins.astype(int).tolist(),
                    energy_per_site=lattice.energy / lattice.sites,
                    magnetisation_per_site=(
                        lattice.extensive()["magnetisation"] / lattice.sites
                    ),
                    acceptance=run.acceptance,
                ).model_dump(),
            }))

        await websocket.send_text(json.dumps({"type": "done"}))
    except WebSocketDisconnect:
        return
    except Exception as error:
        await websocket.send_text(json.dumps({"type": "error", "detail": str(error)}))
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
