# Copyright (C) 2026 Tanguy Marsault - Eigora
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

# ── Discrete measurements ─────────────────────────────────────────────────────

# JSON has no complex type, and amplitudes are complex, so complex arrays
# travel as two real ones: split rather than interleaved, so each half drops
# straight into a typed array on the client.

class ComplexVectorSchema(BaseModel):
    """A complex vector: v[k] = re[k] + 1j * im[k]."""

    re: list[float] = Field(description="Real parts, one per coefficient")
    im: list[float] | None = Field(
        default=None, description="Imaginary parts. Omit for a real vector."
    )

    @model_validator(mode="after")
    def check_shape(self) -> "ComplexVectorSchema":
        if not self.re:
            raise ValueError("the vector must have at least one coefficient")
        if self.im is not None and len(self.im) != len(self.re):
            raise ValueError("re and im must have the same length")
        return self

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return len(self.re)


class ComplexMatrixSchema(BaseModel):
    """A complex square matrix, row-major: M[i][j] = re[i][j] + 1j * im[i][j]."""

    re: list[list[float]] = Field(description="Real parts, row-major")
    im: list[list[float]] | None = Field(
        default=None, description="Imaginary parts, row-major. Omit for a real matrix."
    )

    @model_validator(mode="after")
    def check_shape(self) -> "ComplexMatrixSchema":
        if not self.re:
            raise ValueError("the matrix must have at least one row")
        if any(len(row) != len(self.re) for row in self.re):
            raise ValueError(
                f"the matrix must be square: {len(self.re)} rows, so every row "
                f"must have {len(self.re)} entries"
            )
        if self.im is not None and (
            len(self.im) != len(self.re)
            or any(len(row) != len(self.re) for row in self.im)
        ):
            raise ValueError("im must have the same shape as re")
        return self

    @property
    def dim(self) -> int:
        """Dimension of the Hilbert space."""
        return len(self.re)


class MeasurementRequest(BaseModel):
    """
    A projective measurement of `operator` on `state`.

    The operator must be Hermitian; its eigenvalues are the possible outcomes.
    """

    state: ComplexVectorSchema = Field(
        description="State in the computational basis. Normalised server-side."
    )
    operator: ComplexMatrixSchema = Field(
        description="Hermitian observable in the computational basis"
    )
    n_draws: int | None = Field(
        default=None,
        ge=1,
        le=1_000_000,
        description=(
            "Number of measurements to sample, for showing shot noise against "
            "the exact probabilities. Omit to skip sampling; the outcomes "
            "themselves are the same either way."
        ),
    )
    seed: int | None = Field(
        default=None, ge=0, description="Seed for the draws, to make them reproducible"
    )

    @model_validator(mode="after")
    def check_dimensions(self) -> "MeasurementRequest":
        if self.state.dim != self.operator.dim:
            raise ValueError(
                f"state has {self.state.dim} coefficient(s) but the operator is "
                f"{self.operator.dim}x{self.operator.dim}"
            )
        return self


class OutcomeSchema(BaseModel):
    """
    One distinct eigenvalue, and what measuring it does to the state.

    A degenerate eigenvalue appears once, with its probability summed over the
    whole eigenspace and `state` the projection onto that subspace — not onto
    any single eigenvector.
    """

    value: float = Field(description="The eigenvalue observed")
    probability: float = Field(
        description="Born probability, summed over the eigenspace"
    )
    degeneracy: int = Field(description="Dimension of the eigenspace for this value")
    state: ComplexVectorSchema | None = Field(
        description=(
            "Normalised post-measurement state, None where the probability is "
            "zero and the collapse is undefined."
        )
    )
    count: int | None = Field(
        default=None,
        description="Draws that gave this outcome. None unless n_draws was set.",
    )


class MeasurementResponse(BaseModel):
    outcomes: list[OutcomeSchema] = Field(
        description="One entry per distinct eigenvalue, ascending in value"
    )
    n_draws: int | None = Field(
        default=None, description="Number of draws sampled, echoed back"
    )


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
    view_window: list[float] | None = Field(
        default=None,
        description=(
            "[x_lo, x_hi] to stream, as a sub-range of the grid. The grid has "
            "to be wide enough that the packet never reaches its edge -- the "
            "propagator is FFT-based, so anything that does wraps around and "
            "corrupts the run -- but that same width is mostly empty space "
            "nobody wants to look at. Set this to send only the part worth "
            "drawing. Solving always uses the full grid; this crops the "
            "output. None streams everything."
        ),
    )

    @model_validator(mode="after")
    def check_view_window(self) -> "EvolveRequest":
        if self.view_window is None:
            return self
        if len(self.view_window) != 2:
            raise ValueError("view_window must be [x_lo, x_hi]")
        lo, hi = self.view_window
        if lo >= hi:
            raise ValueError("view_window x_lo must be less than x_hi")
        if hi <= self.grid.x_min or lo >= self.grid.x_max:
            raise ValueError("view_window does not overlap the grid")
        return self


class EvolveFrame(BaseModel):
    frame: int
    t: float
    probability_density: list[float]
    norm: float


class EvolveMetadata(BaseModel):
    x: list[float] = Field(
        description=(
            "Grid points actually being streamed. Equal to the full solver "
            "grid unless view_window cropped it."
        )
    )
    potential: list[float]
    t_max: float
    n_frames: int
    grid_bounds: list[float] = Field(
        default_factory=list,
        description=(
            "[x_min, x_max] of the full grid the equation was solved on, "
            "which is wider than `x` whenever view_window is in play."
        ),
    )
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


# ── Trajectory ────────────────────────────────────────────────────────────────

class TrajectoryRequest(BaseModel):
    grid: GridSchema = Field(default_factory=GridSchema)
    potential: PotentialSchema
    wavepacket: WavepacketSchema = Field(default_factory=WavepacketSchema)
    t_max: float = Field(default=10.0, gt=0, le=50.0)
    dt: float = Field(default=0.01, gt=0, le=0.1)
    n_frames: int = Field(
        default=120,
        ge=10,
        le=400,
        description=(
            "Samples along the trajectory. Can be larger than for /evolve: a "
            "sample here is a handful of numbers, not a full density array."
        ),
    )
    include_classical: bool = Field(
        default=False,
        description=(
            "Also integrate a classical point particle from (x0, k0) in the "
            "same potential and return its path. Rejected for potentials with "
            "a step in them, where a finite-difference force is meaningless."
        ),
    )


class TrajectoryResponse(BaseModel):
    """
    Where the wavepacket is, as a function of time.

    Everything is an expectation value over |psi(x, t)|^2, so this is the
    answer to 'where is the particle' in the only sense quantum mechanics
    allows -- a mean and a spread, not a point.
    """

    times: list[float]
    mean_position: list[float] = Field(description="<x>(t)")
    mean_momentum: list[float] = Field(description="<p>(t)")
    spread_position: list[float] = Field(
        description=(
            "Delta x (t). Constant when the packet is a coherent state, "
            "oscillating at 2*omega otherwise."
        )
    )
    spread_momentum: list[float] = Field(description="Delta p (t)")
    uncertainty_product: list[float] = Field(
        description="Delta x * Delta p (t). Never below 1/2; equal to 1/2 throughout for a coherent state."
    )
    energy: float = Field(description="<H>, conserved over the run")
    boundary_leakage: float = Field(
        description=(
            "Largest fraction of |psi|^2 in the outer 5% of the grid at any "
            "frame. The propagator is FFT-based, so the box is periodic and a "
            "packet reaching one edge reappears at the other -- which turns "
            "every series above into an average over a ring. Above ~1e-3, "
            "widen the grid or shorten t_max; the numbers cannot be trusted."
        )
    )
    classical_position: list[float] | None = Field(
        default=None,
        description=(
            "Position of a classical point particle launched from (x0, k0) in "
            "the same potential, present only when include_classical was set. "
            "In a harmonic well it coincides with mean_position exactly -- "
            "Ehrenfest's theorem, since the force is linear."
        ),
    )
    classical_momentum: list[float] | None = Field(
        default=None,
        description="Present only when include_classical was set.",
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
    "TrajectoryRequest",
    "TrajectoryResponse",
    "SingleAtomStateRequest",
    "SingleAtomStateResponse",
    "ComplexVectorSchema",
    "ComplexMatrixSchema",
    "MeasurementRequest",
    "OutcomeSchema",
    "MeasurementResponse",
]
