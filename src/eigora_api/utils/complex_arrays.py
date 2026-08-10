# Copyright (C) 2026 Tanguy Marsault - Eigora
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Converts between the wire's split re/im arrays and numpy complex arrays.
Keeps the unpacking out of the router.
"""

import numpy as np
from numpy.typing import NDArray

from eigora_api.schemas.qm import ComplexMatrixSchema, ComplexVectorSchema


def to_array(
    schema: ComplexVectorSchema | ComplexMatrixSchema,
) -> NDArray[np.complex128]:
    """
    Join a schema's real and imaginary halves into one complex array.

    The shape follows the schema: (dim,) for a vector, (dim, dim) for a
    matrix. Both halves are rectangular by validation, so this cannot produce
    an object array.
    """
    real = np.asarray(schema.re, dtype=float)
    if schema.im is None:
        return real.astype(complex)
    return real + 1j * np.asarray(schema.im, dtype=float)


def to_vector_schema(values: NDArray[np.complex128]) -> ComplexVectorSchema:
    """Split a complex array of shape (dim,) back into re/im for the wire."""
    return ComplexVectorSchema(re=np.real(values).tolist(), im=np.imag(values).tolist())


__all__ = ["to_array", "to_vector_schema"]
