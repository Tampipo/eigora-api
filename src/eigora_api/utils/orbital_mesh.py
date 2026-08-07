# Copyright (C) 2026 Tanguy Marsault - PhySense
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Server-side isosurface extraction for hydrogen-like orbitals.

Instead of shipping the whole (nx, ny, nz) wavefunction grid to the browser
(tens of MB of JSON) and meshing it client-side, we run marching cubes here and
return only the triangle mesh — a few hundred KB of base64-packed float32/uint32
buffers, ready to drop straight into a WebGL BufferGeometry.
"""

from __future__ import annotations

import base64

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from skimage import measure

# Gaussian blur applied to ψ before meshing, as a fraction of the grid
# resolution (so σ = 0.03·nx ≈ a fixed fraction of the box, independent of the
# resolution slider). This is the key smoothness knob: the 80%-probability shell
# sits far out in ψ's low-gradient tail, where the bare level set wobbles into a
# lumpy potato — blurring by a fixed fraction of the orbital's size flattens that
# wobble into clean lobes while proportional scaling preserves nodal structure.
SIGMA_FRACTION = 0.03125

# Marching-cubes stride. 2 halves the mesh resolution; because the field is
# already blurred smooth this doesn't re-introduce aliasing, and it keeps the
# triangle count (and payload) down.
MC_STEP = 2

# Taubin (λ|μ) mesh smoothing applied to the extracted surface. Removes the
# marching-cubes staircase without the shrinkage plain Laplacian smoothing
# would cause, so lobes come out rounded and symmetric.
TAUBIN_ITERS = 12
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53

# The drawn boundary surface encloses this fraction of the total probability
# (∑ψ²) — the textbook "boundary surface". Picking the isolevel this way rather
# than as a fixed fraction of peak |ψ| auto-adapts to every orbital, so diffuse
# high-n states stay clean shells instead of thin, shredded ones.
ENCLOSED_PROBABILITY = 0.8


def _probability_level(psi: np.ndarray) -> float:
    """|ψ| isolevel enclosing ENCLOSED_PROBABILITY of ∑ψ² (uniform voxels)."""
    sq = np.square(psi, dtype=np.float64).ravel()
    total = float(sq.sum())
    if total <= 0.0:
        return 0.0
    desc = np.sort(sq)[::-1]
    cum = np.cumsum(desc)
    k = int(np.searchsorted(cum, ENCLOSED_PROBABILITY * total))
    k = min(k, desc.size - 1)
    return float(np.sqrt(desc[k]))


def _b64(arr: np.ndarray, dtype: str) -> str:
    """Little-endian raw bytes of ``arr`` as base64 (browser is little-endian)."""
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=dtype).tobytes()
    ).decode("ascii")


def _taubin_smooth(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Taubin (λ|μ) smoothing over the mesh's uniform-weight umbrella operator."""
    V = verts.shape[0]
    # Undirected adjacency (both directions of every triangle edge).
    a = faces[:, [0, 1, 2]].ravel()
    b = faces[:, [1, 2, 0]].ravel()
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    adj = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(V, V)).tocsr()
    adj.data[:] = 1.0  # collapse duplicate edges to weight 1
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    inv_deg = (1.0 / deg)[:, None]

    v = verts.astype(np.float64, copy=True)
    for _ in range(TAUBIN_ITERS):
        for factor in (TAUBIN_LAMBDA, TAUBIN_MU):
            neighbour_avg = adj.dot(v) * inv_deg
            v += factor * (neighbour_avg - v)
    return v


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from face geometry (recomputed post-smooth)."""
    normals = np.zeros_like(verts)
    tris = verts[faces]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    for c in range(3):
        np.add.at(normals, faces[:, c], face_n)
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    return normals / lengths[:, None]


def _lobe(
    field: np.ndarray,
    level: float,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> tuple[dict | None, float]:
    """
    Marching-cubes one signed shell of ``field`` at ``level``.

    Returns the packed lobe (or None if the level isn't crossed) and the largest
    |world coordinate| any vertex reaches, for framing the scene.
    """
    if level <= 0.0 or float(field.max()) <= level:
        return None, 0.0

    try:
        # Lewiner marching cubes: verts in the grid's (spacing-scaled) frame.
        verts, faces, _, _ = measure.marching_cubes(
            field,
            level=level,
            spacing=tuple(spacing.tolist()),
            step_size=MC_STEP,
        )
    except (ValueError, RuntimeError):
        return None, 0.0

    if verts.size == 0 or faces.size == 0:
        return None, 0.0

    # spacing already scaled verts to physical size from index 0; shift to the
    # grid's actual origin, then smooth the surface and recompute normals from
    # the smoothed geometry.
    world = verts + origin
    world = _taubin_smooth(world, faces)
    normals = _vertex_normals(world, faces)
    max_r = float(np.abs(world).max())

    lobe = {
        "positions": _b64(world.ravel(), "<f4"),
        "normals": _b64(normals.ravel(), "<f4"),
        "indices": _b64(faces.ravel(), "<u4"),
        "vertex_count": int(world.shape[0]),
        "triangle_count": int(faces.shape[0]),
    }
    return lobe, max_r


def build_orbital_mesh(
    psi: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> dict:
    """
    Extract the two signed isosurface shells (+ψ / −ψ) of a real orbital.

    ``psi`` has shape (nx, ny, nz) matching axes ``x``, ``y``, ``z``.
    """
    # Blur before meshing (edges clamped, not wrapped). The threshold is then
    # taken on the blurred field so the 80%-probability semantics match what's
    # actually drawn.
    sigma = SIGMA_FRACTION * psi.shape[0]
    psi = gaussian_filter(psi.astype(np.float64), sigma=sigma, mode="nearest")
    level = _probability_level(psi)

    origin = np.array([x[0], y[0], z[0]], dtype=np.float64)
    nx, ny, nz = psi.shape
    spacing = np.array(
        [
            (x[-1] - x[0]) / (nx - 1) if nx > 1 else 1.0,
            (y[-1] - y[0]) / (ny - 1) if ny > 1 else 1.0,
            (z[-1] - z[0]) / (nz - 1) if nz > 1 else 1.0,
        ],
        dtype=np.float64,
    )

    positive, r_pos = _lobe(psi, level, origin, spacing)
    negative, r_neg = _lobe(-psi, level, origin, spacing)

    bound_radius = max(r_pos, r_neg) * 1.1
    if bound_radius <= 0.0:
        bound_radius = 8.0

    return {
        "positive": positive,
        "negative": negative,
        "bound_radius": bound_radius,
    }
