# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Pydantic schemas for the QM router.
Request and response models for eigenstates and time evolution.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── Grid ─────────────────────────────────────────────────────────────────────

class GridSchema(BaseModel):
    x_min: float = Field(default=-10.0)
    x_max: float = Field(default=10.0)
    n_points: int = Field(default=512, ge=64, le=2048)

    @model_validator(mode="after")
    def check_bounds(self) -> "GridSchema":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        return self

class Grid2DSchema(BaseModel):
    x_min: float = Field(default=-10.0)
    x_max: float = Field(default=10.0)
    y_min: float = Field(default=-10.0)
    y_max: float = Field(default=10.0)
    nx: int = Field(default=512, ge=64, le=2048)
    ny: int = Field(default=512, ge=64, le=2048)

    @model_validator(mode="after")
    def check_bounds(self) -> "Grid2DSchema":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be less than y_max")
        return self

class Grid3DSchema(BaseModel):
    x_min: float = Field(default=-10.0)
    x_max: float = Field(default=10.0)
    y_min: float = Field(default=-10.0)
    y_max: float = Field(default=10.0)
    z_min: float = Field(default=-10.0)
    z_max: float = Field(default=10.0)
    nx: int = Field(default=128, ge=32, le=512)
    ny: int = Field(default=128, ge=32, le=512)
    nz: int = Field(default=128, ge=32, le=512)

    @model_validator(mode="after")
    def check_bounds(self) -> "Grid3DSchema":
        if self.x_min >= self.x_max:
            raise ValueError("x_min must be less than x_max")
        if self.y_min >= self.y_max:
            raise ValueError("y_min must be less than y_max")
        if self.z_min >= self.z_max:
            raise ValueError("z_min must be less than z_max")
        return self



# ── Potentials (Option A) ─────────────────────────────────────────────────────

class PotentialType(str, Enum):
    harmonic = "harmonic"
    infinite_well = "infinite_well"
    finite_well = "finite_well"
    barrier = "barrier"
    step = "step"
    double_well = "double_well"
    free = "free"
    custom = "custom"


class HarmonicParams(BaseModel):
    omega: float = Field(default=1.0, gt=0)
    x0: float = Field(default=0.0)


class InfiniteWellParams(BaseModel):
    width: float = Field(default=4.0, gt=0)
    x0: float = Field(default=0.0)


class FiniteWellParams(BaseModel):
    depth: float = Field(default=5.0, gt=0)
    width: float = Field(default=4.0, gt=0)
    x0: float = Field(default=0.0)


class BarrierParams(BaseModel):
    height: float = Field(default=2.0, gt=0)
    width: float = Field(default=1.0, gt=0)
    x0: float = Field(default=0.0)


class StepParams(BaseModel):
    height: float = Field(default=1.0)
    x0: float = Field(default=0.0)


class DoubleWellParams(BaseModel):
    a: float = Field(default=1.0, gt=0)
    b: float = Field(default=4.0, gt=0)


class PotentialSchema(BaseModel):
    type: PotentialType
    params: HarmonicParams | InfiniteWellParams | FiniteWellParams | BarrierParams | StepParams | DoubleWellParams | None = None
    # Option B: custom V(x) values on the grid
    values: list[float] | None = Field(default=None, description="Custom V(x) values, one per grid point (Option B)")

    @model_validator(mode="after")
    def check_custom(self) -> "PotentialSchema":
        if self.type == PotentialType.custom and self.values is None:
            raise ValueError("values must be provided for custom potential")
        if self.type != PotentialType.custom and self.values is not None:
            raise ValueError("values should only be provided for custom potential")
        return self




# ── Eigenstates ───────────────────────────────────────────────────────────────

class EigenstatesRequest(BaseModel):
    grid: GridSchema = Field(default_factory=GridSchema)
    potential: PotentialSchema
    n_states: int = Field(default=6, ge=1, le=20)


class EigenstatesResponse(BaseModel):
    x: list[float]
    potential: list[float]
    energies: list[float]
    wavefunctions: list[list[float]]  # shape: (n_states, n_points)
    n_states: int

# ── Separable 2D states ───────────────────────────────────────────────────────

class SeparableStateRequest(BaseModel):
    """
    One eigenstate of a system that separates along x and y.

    The potential is given per axis, so a 2D well is `infinite_well` on both
    axes, an anisotropic trap is `harmonic` on both with different omega, and
    the two axes may differ freely.
    """

    grid: Grid2DSchema = Field(default_factory=Grid2DSchema)
    potential_x: PotentialSchema
    potential_y: PotentialSchema
    n1: int = Field(
        default=0,
        ge=0,
        le=99,
        description=(
            "Index of the state along x, 0 being the ground state. This is a "
            "position in the ascending list of states, not the physical "
            "quantum number -- those differ per potential (a box starts at "
            "n=1, an oscillator at n=0) and are returned in `label`."
        ),
    )
    n2: int = Field(default=0, ge=0, le=99, description="Index of the state along y.")
    n_states: int = Field(
        default=20,
        ge=1,
        le=100,
        description="States computed per axis that has no analytic solution.",
    )


class SeparableStateResponse(BaseModel):
    """
    The state as its two one-dimensional factors: psi(x, y) = psi_x(x) * psi_y(y).

    Sending the factors rather than the (nx, ny) field keeps the payload
    proportional to nx + ny instead of nx * ny; the client reconstructs the
    field with an outer product.
    """

    x: list[float]
    y: list[float]
    psi_x: list[float] = Field(description="Normalised factor along x")
    psi_y: list[float] = Field(description="Normalised factor along y")
    potential_x: list[float] = Field(description="V(x) sampled on the x axis")
    potential_y: list[float] = Field(description="V(y) sampled on the y axis")
    energy: float
    energy_x: float
    energy_y: float
    label: list[int] = Field(description="Physical quantum numbers of the state")
    quantum_numbers: list[str] = Field(description="Names of those quantum numbers")
    degeneracy: int = Field(description="How many states share this energy")
    is_exact: bool = Field(
        description="False if either axis had to be solved numerically"
    )


class SingleAtomStateRequest(BaseModel):
    grid: Grid3DSchema = Field(default_factory=Grid3DSchema)
    Z: int = Field(default=1, ge=1, le=100, description="Atomic number")
    n: int = Field(default=1, ge=1, le=10, description="Principal quantum number")
    l: int = Field(default=0, ge=0, le=9, description="Orbital angular momentum quantum number")
    m: int = Field(default=0, description="Magnetic quantum number")

class LobeMeshSchema(BaseModel):
    """
    One signed isosurface shell as a triangle mesh, packed for direct upload to
    a WebGL BufferGeometry. Each string is base64 of the little-endian raw
    bytes: positions/normals are flat float32 [x,y,z,...]; indices are flat
    uint32 triangle vertex indices.
    """

    positions: str
    normals: str
    indices: str
    vertex_count: int
    triangle_count: int


class SingleAtomStateResponse(BaseModel):
    # Server-side isosurface: the ±ψ boundary shells at the 80%-probability
    # level, instead of the full (nx, ny, nz) field — a few hundred KB rather
    # than tens of MB, and no client-side meshing.
    positive: LobeMeshSchema | None
    negative: LobeMeshSchema | None
    bound_radius: float  # half-size of the shape's bounding cube (Bohr radii)
    Z: int
    n: int
    l: int
    m: int
    
# ── Evolution (WebSocket) ─────────────────────────────────────────────────────

class WavepacketSchema(BaseModel):
    x0: float = Field(default=-5.0, description="Initial position")
    k0: float = Field(default=1.5, description="Initial momentum")
    sigma: float = Field(default=1.0, gt=0, description="Spatial width")


class EvolveRequest(BaseModel):
    grid: GridSchema = Field(default_factory=GridSchema)
    potential: PotentialSchema
    wavepacket: WavepacketSchema = Field(default_factory=WavepacketSchema)
    t_max: float = Field(default=10.0, gt=0, le=50.0)
    dt: float = Field(default=0.01, gt=0, le=0.1)
    n_frames: int = Field(default=60, ge=10, le=200)


class EvolveFrame(BaseModel):
    frame: int
    t: float
    probability_density: list[float]
    norm: float


class EvolveMetadata(BaseModel):
    x: list[float]
    potential: list[float]
    t_max: float
    n_frames: int
    predicted_transmission: float | None = Field(
        default=None,
        description=(
            "Energy-averaged transmission for a barrier potential: the "
            "integral of |phi(k)|^2 * T(E(k)) dk over the wavepacket's "
            "momentum distribution. None for non-barrier potentials."
        ),
    )
    mean_energy_transmission: float | None = Field(
        default=None,
        description=(
            "T evaluated at a single energy, E = k0^2/2 -- the naive "
            "prediction a plane wave at the wavepacket's mean momentum "
            "would give, for comparison against predicted_transmission. "
            "None for non-barrier potentials."
        ),
    )


__all__ = [
    "GridSchema",
    "Grid2DSchema",
    "Grid3DSchema",
    "PotentialType",
    "PotentialSchema",
    "EigenstatesRequest",
    "EigenstatesResponse",
    "SeparableStateRequest",
    "SeparableStateResponse",
    "WavepacketSchema",
    "EvolveRequest",
    "EvolveFrame",
    "EvolveMetadata",
    "SingleAtomStateRequest",
    "SingleAtomStateResponse",
]
