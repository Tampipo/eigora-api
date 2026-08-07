# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Maps PotentialSchema to eigora.qm Potential objects.
Keeps routing logic out of the router.
"""

import numpy as np

from eigora.qm.potentials import (
    Potential,
    FreeParticle,
    HarmonicWell,
    InfiniteSquareWell,
    FiniteSquareWell,
    RectangularBarrier,
    PotentialStep,
    DoubleWell,
)
from eigora.grids import GridND

from eigora_api.schemas.qm import PotentialSchema, PotentialType


class CustomPotential(Potential):
    """Wraps a pre-evaluated V(x) array as a Potential object."""

    def __init__(self, values: list[float]) -> None:
        self._values = np.array(values, dtype=np.float64)

    def __call__(self, x):
        if len(x) != len(self._values):
            raise ValueError(
                f"Custom potential has {len(self._values)} values "
                f"but grid has {len(x)} points"
            )
        return self._values


def build_potential(schema: PotentialSchema, grid: GridND) -> Potential:
    """
    Construct a eigora.qm Potential from a PotentialSchema.

    Parameters
    ----------
    schema : PotentialSchema
    grid : GridND
        Needed to validate custom potential size.

    Returns
    -------
    Potential
    """
    p = schema.params

    match schema.type:
        case PotentialType.free:
            return FreeParticle()

        case PotentialType.harmonic:
            kwargs = p.model_dump() if p else {}
            return HarmonicWell(**kwargs)

        case PotentialType.infinite_well:
            kwargs = p.model_dump() if p else {}
            return InfiniteSquareWell(**kwargs)

        case PotentialType.finite_well:
            kwargs = p.model_dump() if p else {}
            return FiniteSquareWell(**kwargs)

        case PotentialType.barrier:
            kwargs = p.model_dump() if p else {}
            return RectangularBarrier(**kwargs)

        case PotentialType.step:
            kwargs = p.model_dump() if p else {}
            return PotentialStep(**kwargs)

        case PotentialType.double_well:
            kwargs = p.model_dump() if p else {}
            return DoubleWell(**kwargs)

        case PotentialType.custom:
            if len(schema.values) != grid.n_points:
                raise ValueError(
                    f"Custom potential has {len(schema.values)} values "
                    f"but grid has {grid.n_points} points"
                )
            return CustomPotential(schema.values)


__all__ = ["build_potential", "CustomPotential"]
