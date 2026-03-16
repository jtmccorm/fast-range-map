# Maritime Reach Map Generator

`maritime_reach_map.py` generates a single static PNG visualizing maritime operational reach or theater sustainment throughput from one or more hubs while treating land as an impassable barrier.

## Setup

The repo already includes the Natural Earth land dataset under `data/ne_10m_land/`, so a fresh workstation only needs Python and the packages in `requirements.txt`.

Recommended clean install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 maritime_reach_map.py
```

Alternative install matching the local setup used during development:

```bash
pip install --target .vendor -r requirements.txt
python3 maritime_reach_map.py
```

Notes:

- The script automatically adds `.vendor/` to `sys.path` if that folder exists.
- The script also sets a local Matplotlib config/cache directory under `.mplconfig/`, so no extra GUI or desktop setup is required.
- Verified locally with Homebrew `python3` on macOS and the package set in `requirements.txt`.

## Usage

Run the default example from the prompt:

```bash
python3 maritime_reach_map.py
```

Specify your own hubs and range:

```bash
python3 maritime_reach_map.py \
  --hub 14.829 120.283 \
  --hub -12.400 130.800 \
  --range-nm 2000 \
  --output maritime_reach_map.png
```

Generate a throughput-capacity map from cached distance fields and vessel specs:

```bash
python3 maritime_reach_map.py \
  --output-mode throughput \
  --hub 14.829 120.283 \
  --hub 13.444 144.657 \
  --hub-vessel 1 100 16 2000 \
  --hub-vessel 1 50 22 1500 \
  --hub-vessel 2 80 15 2600 \
  --min-cycle-days 1.0 \
  --throughput-contours 5 10 25 50 75 100 \
  --output ./output/maritime_throughput_map.png    
```

`--hub-vessel` takes `HUB_INDEX PAYLOAD_TONS SPEED_KNOTS RANGE_NM` and can be repeated to model multiple vessels or vessel types at the same hub.
`--min-cycle-days` sets the minimum delivery cycle time assumed for every vessel in throughput mode, which caps each vessel at `payload_tons / min_cycle_days` tons/day near the hub.

## Benchmarking

In order to test the script on your machine, run the grid-resolution benchmark with:

```bash
python3 benchmark.py
```

This writes benchmark PNGs and a summary CSV under `benchmarks/`. The CSV includes `step_km`, approximate `grid_cells`, elapsed runtime in seconds, and peak memory in KB for each tested grid resolution.

Grid resolution can greatly affect the routes available in archipelagos (such as the Philipines) altering the estimated distance travelled. This feature allows you to balance accuracy vs runtime in developing graphics.

## Notes

- The script uses bundled Natural Earth 1:10m land polygons in `data/ne_10m_land/`.
- Reach is computed with a water-routed cost-distance grid, so paths can bend around coastlines and islands instead of stopping at first landfall.
- Throughput mode reuses the cached `.npy` distance fields, applies a `1 / max(distance, d_min)` delivery model, caps each vessel at `payload_tons / min_cycle_days` tons/day, and zeros vessel contributions beyond half of each vessel's listed range.
- Hubs are rendered at the exact input coordinates. If a hub falls in a land cell, the internal routing origin is snapped to the nearest water cell while keeping the visible marker at the original hub location.
- `--step-km` controls the routing-grid resolution. Smaller values improve channel fidelity but increase runtime.
- `--output-mode range` preserves the original reach visualization. `--output-mode throughput` renders a tons/day heatmap plus labeled sustainment contours.
- `--rays` is deprecated and ignored; it remains accepted only for backward compatibility with earlier versions.
- Default map bounds are `70E to 170E` and `20S to 40N`.
