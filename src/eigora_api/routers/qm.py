# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
QM router.

POST /qm/eigenstates  — solve time-independent Schrödinger equation
WS   /qm/evolve       — stream time evolution frames via WebSocket
"""

import json

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from physense_utils.grids import GridND
from physense_qm import QuantumSystem1D
from physense_qm.states.wavepacket import GaussianWavepacket
from physense_qm.potentials import RectangularBarrier, SeparablePotential
from physense_qm.scattering import energy_averaged_transmission
from physense_qm.spectra import spectrum_for
from physense_qm.states.orbitals import SingleAtomState
from physense_api.schemas.qm import (
    EigenstatesRequest,
    EigenstatesResponse,
    EvolveRequest,
    EvolveFrame,
    EvolveMetadata,
    SeparableStateRequest,
    SeparableStateResponse,
    SingleAtomStateResponse,
    SingleAtomStateRequest,
)
from physense_api.utils.potentials import build_potential
from physense_api.utils.orbital_mesh import build_orbital_mesh

router = APIRouter(prefix="/qm", tags=["Quantum Mechanics"])


@router.post("/eigenstates", response_model=EigenstatesResponse)
def eigenstates(req: EigenstatesRequest) -> EigenstatesResponse:
    """
    Solve the time-independent Schrödinger equation and return eigenstates.

    Accepts either a named potential (Option A) or a custom V(x) array (Option B).
    """
    grid = GridND.line(req.grid.x_min, req.grid.x_max, req.grid.n_points)
    try:
        potential = build_potential(req.potential, grid)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    system = QuantumSystem1D(grid=grid, potential=potential)
    sol = system.solve(n_states=req.n_states)

    return EigenstatesResponse(
        x=grid.x.tolist(),
        potential=sol.potential.tolist(),
        energies=sol.energies.tolist(),
        wavefunctions=[psi.tolist() for psi in sol.wavefunctions],
        n_states=sol.n_states,
    )

@router.post("/separable-state", response_model=SeparableStateResponse)
def separable_state(req: SeparableStateRequest) -> SeparableStateResponse:
    """
    One eigenstate of a system that separates along x and y.

    The potential is given per axis, so `infinite_well` on both axes is a
    particle in a 2D box, `harmonic` on both is a 2D trap, and the axes may
    differ. Each axis is solved analytically when its potential has a known
    solution and numerically otherwise; the state is the product of the two.
    """
    grid = GridND.uniform(
        bounds=[
            (req.grid.x_min, req.grid.x_max),
            (req.grid.y_min, req.grid.y_max),
        ],
        shape=(req.grid.nx, req.grid.ny),
    )
    x_grid, y_grid = grid.sub(0, 1), grid.sub(1, 2)

    try:
        potential_x = build_potential(req.potential_x, x_grid)
        potential_y = build_potential(req.potential_y, y_grid)
        spectrum = spectrum_for(
            SeparablePotential([potential_x, potential_y]),
            grid=grid,
            n_states=req.n_states,
        )
        label_x = _state_at(spectrum.blocks[0], req.n1, "n1")
        label_y = _state_at(spectrum.blocks[1], req.n2, "n2")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    x, y = x_grid.x, y_grid.x
    label = label_x + label_y

    return SeparableStateResponse(
        x=x.tolist(),
        y=y.tolist(),
        psi_x=spectrum.blocks[0].wavefunction(label_x)(x).tolist(),
        psi_y=spectrum.blocks[1].wavefunction(label_y)(y).tolist(),
        potential_x=np.asarray(potential_x(x)).tolist(),
        potential_y=np.asarray(potential_y(y)).tolist(),
        energy=spectrum.energy(label),
        energy_x=spectrum.blocks[0].energy(label_x),
        energy_y=spectrum.blocks[1].energy(label_y),
        label=list(label),
        quantum_numbers=list(spectrum.quantum_numbers),
        degeneracy=spectrum.degeneracy(label),
        is_exact=spectrum.is_exact,
    )


def _state_at(block, index: int, field: str) -> tuple[int, ...]:
    """The block's `index`-th state, as physical quantum numbers."""
    states = block.states(index + 1)
    if index >= len(states):
        raise ValueError(
            f"{field}={index} is out of range: only {len(states)} state(s) "
            f"are available on that axis"
        )
    return states[index]


@router.post("/single-atom-state", response_model=SingleAtomStateResponse)
def single_atom_state(req: SingleAtomStateRequest) -> SingleAtomStateResponse:
    """
    Compute the single-atom state for a given potential and quantum numbers.

    Accepts either a named potential (Option A) or a custom V(x) array (Option B).
    """
    grid = GridND.uniform(
        bounds=[
            (req.grid.x_min, req.grid.x_max),
            (req.grid.y_min, req.grid.y_max),
            (req.grid.z_min, req.grid.z_max),
        ],
        shape=(req.grid.nx, req.grid.ny, req.grid.nz),
    )

    try:
        atom_state = SingleAtomState(Z=req.Z, n=req.n, l=req.l, m=req.m)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    psi_values = np.asarray(atom_state.wavefunction_on_grid(grid))

    mesh = build_orbital_mesh(psi_values, grid.values(0), grid.values(1), grid.values(2))

    return SingleAtomStateResponse(
        positive=mesh["positive"],
        negative=mesh["negative"],
        bound_radius=mesh["bound_radius"],
        Z=req.Z,
        n=req.n,
        l=req.l,
        m=req.m,
    )

@router.websocket("/evolve")
async def evolve(websocket: WebSocket) -> None:
    """
    Stream time evolution frames via WebSocket.

    Protocol:
    1. Client connects and sends EvolveRequest as JSON
    2. Server sends metadata frame first: { "type": "metadata", ... }
    3. Server streams evolution frames: { "type": "frame", "frame": i, "t": t, ... }
    4. Server sends done signal: { "type": "done" }
    """
    await websocket.accept()

    try:
        data = await websocket.receive_text()
        req = EvolveRequest.model_validate_json(data)

        grid = GridND.line(req.grid.x_min, req.grid.x_max, req.grid.n_points)
        potential = build_potential(req.potential, grid)
        system = QuantumSystem1D(grid=grid, potential=potential)

        wavepacket = GaussianWavepacket(
            x0=req.wavepacket.x0,
            k0=req.wavepacket.k0,
            sigma=req.wavepacket.sigma,
        )

        predicted_transmission = None
        mean_energy_transmission = None
        if isinstance(potential, RectangularBarrier):
            predicted_transmission = energy_averaged_transmission(potential, wavepacket)
            mean_energy = 0.5 * wavepacket.k0**2
            mean_energy_transmission = potential.transmission_coefficient(mean_energy)

        # Send metadata first so frontend can set up the canvas
        metadata = EvolveMetadata(
            x=grid.x.tolist(),
            potential=potential(grid.x).tolist(),
            t_max=req.t_max,
            n_frames=req.n_frames,
            predicted_transmission=predicted_transmission,
            mean_energy_transmission=mean_energy_transmission,
        )
        await websocket.send_text(json.dumps({"type": "metadata", **metadata.model_dump()}))

        # Run evolution and stream frames
        evo = system.evolve(
            initial_state=wavepacket,
            t_max=req.t_max,
            dt=req.dt,
            n_frames=req.n_frames,
        )

        for i in range(evo.n_frames):
            prob = np.abs(evo.psi[i]) ** 2
            norm = float(np.trapezoid(prob, grid.x))
            frame = EvolveFrame(
                frame=i,
                t=float(evo.times[i]),
                probability_density=prob.tolist(),
                norm=norm,
            )
            await websocket.send_text(json.dumps({"type": "frame", **frame.model_dump()}))

        await websocket.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass



__all__ = ["router"]
