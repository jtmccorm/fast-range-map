# Maritime Reach Map Generator

`maritime_reach_map.py` generates a single static PNG visualizing maritime operational reach from one or more hubs while treating land as an impassable barrier.

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

## Notes

- The script uses bundled Natural Earth 1:10m land polygons in `data/ne_10m_land/`.
- Reach is computed with a water-routed cost-distance grid, so paths can bend around coastlines and islands instead of stopping at first landfall.
- Hubs are rendered at the exact input coordinates. If a hub falls in a land cell, the internal routing origin is snapped to the nearest water cell while keeping the visible marker at the original hub location.
- `--step-km` controls the routing-grid resolution. Smaller values improve channel fidelity but increase runtime.
- `--rays` is deprecated and ignored; it remains accepted only for backward compatibility with earlier versions.
- Default map bounds are `70E to 170E` and `20S to 40N`.
