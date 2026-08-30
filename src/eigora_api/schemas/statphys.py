# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pydantic schemas for the statistical physics router.

Units follow the library: k_B = hbar = 1, so a temperature *is* an energy and
entropy is dimensionless. The one exception is `coexistence`, which needs a
real mass to fix the thermal wavelength and therefore reports pressure in
pascals; its energies are quoted as temperatures in kelvin.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SystemType(str, Enum):
    two_level = "two_level"
    harmonic = "harmonic"
    spin = "spin"
    rotor = "rotor"
    ideal_gas = "ideal_gas"


class EnsembleType(str, Enum):
    canonical = "canonical"
    magnetic = "magnetic"
    isobaric = "isobaric"


class SystemSchema(BaseModel):
    """
    One subsystem, optionally repeated.

    `copies` is a tensor product of *distinguishable* copies, so N spins are
    `spin` with `copies = N`. Indistinguishable particles are a different
    construction and are not reachable from here.
    """

    type: SystemType = SystemType.two_level
    splitting: float = Field(default=1.0, gt=0, description="two_level gap")
    omega: float = Field(default=1.0, gt=0, description="harmonic frequency")
    spin: float = Field(default=0.5, ge=0, description="spin quantum number j")
    rotational_constant: float = Field(default=1.0, gt=0, description="rotor b")
    particles: int = Field(default=1, ge=0, description="ideal_gas particle count")
    volume: float = Field(default=100.0, gt=0, description="ideal_gas volume")
    copies: int = Field(default=1, ge=1, le=5000)


class EnsembleSchema(BaseModel):
    type: EnsembleType = EnsembleType.canonical
    magnetic_field: float = Field(default=0.0, description="h, magnetic ensemble")
    pressure: float = Field(default=1.0, gt=0, description="P, isobaric ensemble")


class TemperatureRange(BaseModel):
    minimum: float = Field(default=0.1, gt=0)
    maximum: float = Field(default=10.0, gt=0)
    points: int = Field(default=80, ge=2, le=400)
    logarithmic: bool = True

    @model_validator(mode="after")
    def check_order(self) -> "TemperatureRange":
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be less than maximum")
        return self


class ThermodynamicsRequest(BaseModel):
    system: SystemSchema = Field(default_factory=SystemSchema)
    ensemble: EnsembleSchema = Field(default_factory=EnsembleSchema)
    temperatures: TemperatureRange = Field(default_factory=TemperatureRange)


class ThermodynamicsResponse(BaseModel):
    """
    Everything one sweep yields, per temperature.

    `potential_name` says which thermodynamic potential `potential` is: the
    Helmholtz free energy canonically, the Gibbs energy once volume is free,
    and so on. Both are floats, so reporting one as the other is an error
    nothing downstream would catch.
    """

    temperatures: list[float]
    energy: list[float]
    heat_capacity: list[float]
    entropy: list[float]
    potential: list[float]
    potential_name: str
    magnetisation: list[float] | None = None
    susceptibility: list[float] | None = None
    volume: list[float] | None = None
    states: int | None = Field(
        default=None, description="Total microstates, null if unbounded"
    )


class CoexistenceRequest(BaseModel):
    """
    Solid-gas equilibrium: an Einstein crystal against an ideal gas.

    Defaults are argon: `hbar omega / k_B = 90 K`, and a cohesion energy equal
    to the measured sublimation enthalpy plus the zero point the atom is given
    back, since the slope of `log P` against `1/T` is `eps_c - 3 hbar omega/2`.
    """

    omega: float = Field(default=90.0, gt=0, description="hbar omega / k_B, kelvin")
    cohesion: float = Field(default=1066.0, gt=0, description="eps_c / k_B, kelvin")
    mass_amu: float = Field(default=39.948, gt=0)
    temperatures: TemperatureRange = Field(
        default_factory=lambda: TemperatureRange(
            minimum=45.0, maximum=80.0, points=80, logarithmic=False
        )
    )


class CoexistenceResponse(BaseModel):
    temperatures: list[float]
    chemical_potential: list[float] = Field(description="mu_c, kelvin")
    pressure: list[float] = Field(description="sublimation pressure, pascals")
    latent_heat: float = Field(description="eps_c - 3 hbar omega / 2, kelvin")


class GasRequest(BaseModel):
    """
    A Monte Carlo gas of spin-`s` particles in a periodic cubic box.

    The spin fixes both halves: `2s+1` states per orbital, and half-integer
    spin means fermions while integer spin means bosons. That is the
    spin-statistics theorem doing the dispatch rather than the caller being
    asked the same fact twice.
    """

    box: float = Field(default=5.0, gt=0, description="box edge, nanometres")
    particles: int = Field(default=40, ge=1, le=400)
    spin: float = Field(default=0.5, ge=0, le=3)
    temperature: float = Field(default=1000.0, gt=0, description="kelvin")
    samples: int = Field(default=1500, ge=100, le=20000)
    burn_in_sweeps: int = Field(default=200, ge=0, le=5000)
    seed: int | None = None

    @model_validator(mode="after")
    def check_spin(self) -> "GasRequest":
        if abs(2.0 * self.spin - round(2.0 * self.spin)) > 1e-9:
            raise ValueError("spin must be a multiple of 1/2")
        return self


class GasResponse(BaseModel):
    """
    Sampled occupations, and the closed form they are checked against.

    `chemical_potential_fitted` is what the simulation alone can give: a least
    squares fit of the distribution to the measured occupations. It is
    reported beside the exact value so the two can be compared rather than one
    standing in for the other.
    """

    energies: list[float] = Field(description="distinct orbital energies, kelvin")
    occupation: list[float]
    occupation_error: list[float]
    distribution: list[float] = Field(description="closed form at the exact mu")
    statistics: str
    fermi_temperature: float
    chemical_potential_fitted: float
    chemical_potential_error: float
    chemical_potential_exact: float
    orbitals: int
    acceptance: float


class IsingRequest(BaseModel):
    """A lattice to stream. `size` is one edge; the lattice is square."""

    size: int = Field(default=48, ge=4, le=128)
    coupling: float = Field(default=1.0)
    temperature: float = Field(default=2.269, gt=0)
    sweeps_per_frame: int = Field(default=2, ge=1, le=50)
    frames: int = Field(default=200, ge=1, le=2000)
    seed: int | None = None


class IsingMetadata(BaseModel):
    size: int
    critical_temperature: float = Field(
        description="2 / log(1 + sqrt(2)), exact for the square lattice"
    )
    sites: int
    frames: int
    sweeps_per_frame: int


class IsingFrame(BaseModel):
    frame: int
    sweep: int
    temperature: float
    spins: list[list[int]]
    energy_per_site: float
    magnetisation_per_site: float
    acceptance: float


class GasStreamRequest(GasRequest):
    """
    One chain, streamed as it converges.

    `frames` splits the run into that many reports, so the client watches the
    distribution form rather than waiting for the answer. That convergence is
    the demonstration; the converged number is what `gas-sweep` is for.
    """

    frames: int = Field(default=60, ge=2, le=500)


class GasMetadata(BaseModel):
    energies: list[float]
    degeneracies: list[int]
    statistics: str
    fermi_temperature: float
    chemical_potential_exact: float
    distribution: list[float] = Field(description="closed form at the exact mu")
    orbitals: int
    particles: int
    temperature: float
    frames: int


class GasStreamFrame(BaseModel):
    """
    A running average, not a snapshot.

    `occupation` is the mean over every sample so far, so the error bars close
    as the chain proceeds. `chemical_potential_fitted` is null early on, when
    the fit has nothing stable to bite on.
    """

    frame: int
    samples: int
    occupation: list[float]
    occupation_error: list[float]
    chemical_potential_fitted: float | None = None
    chemical_potential_error: float | None = None
    acceptance: float


class GasSweepRequest(BaseModel):
    """
    Many converged chains, one per value of the swept parameter.

    `vary` picks which axis: temperature at fixed particle number, or particle
    number at fixed temperature. Points are independent, so they run in
    parallel processes.
    """

    box: float = Field(default=5.0, gt=0, description="box edge, nanometres")
    spin: float = Field(default=0.5, ge=0, le=3)
    vary: Literal["temperature", "particles"] = "temperature"
    values: list[float] = Field(min_length=1, max_length=24)
    particles: int = Field(default=40, ge=1, le=400, description="fixed when varying T")
    temperature: float = Field(default=1000.0, gt=0, description="fixed when varying N")
    samples: int = Field(default=800, ge=100, le=8000)
    burn_in_sweeps: int = Field(default=150, ge=0, le=5000)
    seed: int | None = None

    @model_validator(mode="after")
    def check_values(self) -> "GasSweepRequest":
        if any(value <= 0 for value in self.values):
            raise ValueError("every swept value must be positive")
        if self.vary == "particles" and any(
            value != int(value) for value in self.values
        ):
            raise ValueError("particle numbers must be whole")
        if abs(2.0 * self.spin - round(2.0 * self.spin)) > 1e-9:
            raise ValueError("spin must be a multiple of 1/2")
        return self


class GasSweepPoint(BaseModel):
    temperature: float
    particles: int
    fermi_temperature: float
    chemical_potential_fitted: float
    chemical_potential_error: float
    chemical_potential_exact: float
    sommerfeld: float | None = Field(
        default=None, description="Ef[1 - (pi^2/12)(T/Tf)^2], fermions only"
    )
    maxwell_boltzmann: float | None = Field(
        default=None, description="the classical limit, fermions only"
    )


class GasSweepResponse(BaseModel):
    vary: str
    statistics: str
    points: list[GasSweepPoint]
