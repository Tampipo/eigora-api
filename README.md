<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="Eigora" src="assets/logo-light.svg" width="240">
  </picture>
</p>

# eigora-api

FastAPI backend for the [Eigora](https://eigora.tampipo.fr) platform. Exposes physics simulations from `eigora` (and future modules) as an HTTP/WebSocket API.

Live at: **https://api.eigora.tampipo.fr**  
Interactive docs: **https://api.eigora.tampipo.fr/docs**

---

## Structure

```
src/eigora_api/
  main.py              # FastAPI app, CORS, lifespan
  routers/
    qm.py              # /v1/qm/eigenstates, /v1/qm/separable-state,
                       # /v1/qm/single-atom-state (POST), /v1/qm/evolve (WebSocket)
  schemas/
    qm.py              # Pydantic request/response models
  utils/
    potentials.py      # Maps PotentialSchema → eigora.qm Potential objects
```

---

## Running locally

```bash
git clone https://github.com/Tampipo/eigora-api
cd eigora-api
pip install -e ".[dev]"
uvicorn eigora_api.main:app --reload --port 8000
```

CORS is configured via the `ALLOWED_ORIGINS` environment variable (comma-separated). Defaults to `http://localhost:3000` for local development.

```bash
ALLOWED_ORIGINS=http://localhost:3000,https://eigora.tampipo.fr uvicorn eigora_api.main:app
```

---

## Running with Docker

```bash
docker build -t eigora-api .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=http://localhost:3000 eigora-api
```

---

## API reference

### `POST /v1/qm/eigenstates`

Solves the time-independent Schrödinger equation and returns eigenstates.

**Request:**

```json
{
  "grid": { "x_min": -8, "x_max": 8, "n_points": 512 },
  "potential": {
    "type": "harmonic",
    "params": { "omega": 1.0, "x0": 0.0 }
  },
  "n_states": 5
}
```

**Potential types (Option A):** `free`, `harmonic`, `infinite_well`, `finite_well`, `barrier`, `step`, `double_well`

**Custom potential (Option B):** pass `type: "custom"` and a `values` array of length `n_points` :

```json
{
  "potential": {
    "type": "custom",
    "values": [0.0, 0.1, 0.5, ...]
  }
}
```

**Response:**

```json
{
  "x": [...],
  "potential": [...],
  "energies": [0.5, 1.5, 2.5, 3.5, 4.5],
  "wavefunctions": [[...], [...], ...],
  "n_states": 5
}
```

---

### `POST /v1/qm/separable-state`

Returns one eigenstate of a system that separates along x and y — a particle in
a well in both directions, a 2D trap, or any pair of 1D potentials. Each axis is
solved analytically when its potential has a known solution (`harmonic`,
`infinite_well`) and numerically otherwise.

**Request:**

```json
{
  "grid": { "x_min": 0, "x_max": 6, "y_min": 0, "y_max": 3, "nx": 128, "ny": 128 },
  "potential_x": { "type": "infinite_well", "params": { "width": 6.0, "x0": 3.0 } },
  "potential_y": { "type": "infinite_well", "params": { "width": 3.0, "x0": 1.5 } },
  "n1": 2,
  "n2": 3,
  "n_states": 20
}
```

`n1` and `n2` are **positions in the ascending list of states**, 0 being the
ground state — not the physical quantum numbers, which differ per potential (a
box starts at n=1, an oscillator at n=0) and come back in `label`.

**Response:** the state as its two 1D factors, so the payload grows like
`nx + ny` rather than `nx * ny`. Rebuild the field with an outer product:
`psi[i][j] = psi_x[i] * psi_y[j]`.

```json
{
  "x": [...], "y": [...],
  "psi_x": [...], "psi_y": [...],
  "potential_x": [...], "potential_y": [...],
  "energy": 10.0067, "energy_x": 1.2337, "energy_y": 8.773,
  "label": [3, 4],
  "quantum_numbers": ["n_1", "n_2"],
  "degeneracy": 1,
  "is_exact": true
}
```

---

### `WS /v1/qm/evolve`

Streams wavepacket time evolution frames.

**Protocol:**

1. Client connects to `ws://host/v1/qm/evolve`
2. Client sends `EvolveRequest` as JSON
3. Server sends `{ "type": "metadata", "x": [...], "potential": [...], ... }`
4. Server streams `{ "type": "frame", "frame": i, "t": 0.1, "probability_density": [...], "norm": 1.0 }`
5. Server sends `{ "type": "done" }`

**Request:**

```json
{
  "grid": { "x_min": -20, "x_max": 20, "n_points": 512 },
  "potential": { "type": "barrier", "params": { "height": 2.0, "width": 2.0 } },
  "wavepacket": { "x0": -8.0, "k0": 1.5, "sigma": 1.5 },
  "t_max": 10.0,
  "dt": 0.005,
  "n_frames": 80
}
```

**Example (Python):**

```python
import asyncio, websockets, json

async def test():
    async with websockets.connect("wss://api.eigora.tampipo.fr/v1/qm/evolve") as ws:
        await ws.send(json.dumps({
            "grid": {"x_min": -20, "x_max": 20, "n_points": 512},
            "potential": {"type": "barrier", "params": {"height": 2.0, "width": 2.0}},
            "wavepacket": {"x0": -8.0, "k0": 1.5, "sigma": 1.5},
            "t_max": 10.0, "dt": 0.005, "n_frames": 60,
        }))
        async for msg in ws:
            data = json.loads(msg)
            print(data["type"], data.get("t", ""))
            if data["type"] == "done":
                break

asyncio.run(test())
```

---

## Limits

| Parameter | Min | Max |
|---|---|---|
| `n_points` | 64 | 2048 |
| `n_states` | 1 | 20 |
| `n_frames` | 10 | 200 |
| `t_max` | — | 50.0 |
| `dt` | — | 0.1 |

---

## Exporting the OpenAPI schema

```bash
pip install pyyaml
python scripts/export_openapi.py
```

Outputs `openapi.yaml` at the project root, consumed by `orval` in the frontend to generate TypeScript clients.

---

## Running tests

```bash
pytest
```

---

## Deployment

The API is containerised and deployed on a k3s cluster via ArgoCD. The CI pipeline (GitHub Actions) builds and pushes a multi-arch Docker image to GHCR on every push to `main`. ArgoCD Image Updater automatically bumps the image tag in the gitops repo and triggers a rollout.

```
push to main → CI → ghcr.io/tampipo/eigora-api:x.y.z → ArgoCD → k3s
```

---

## Dependencies

- `fastapi >= 0.115`
- `uvicorn >= 0.30`
- `websockets >= 13.0`
- `eigora`
- `eigora`
