# Research Brief #37 — Precision Humanitarian Drop (Digital Twin)

**Date:** 2026-09-15
**Project:** AIR Lab (Nexus-Predator) — rescue-drone line (docs/rescue-drone/05)
**Status:** implemented, tested, demoed.  Program priority #13.

---

## 1. Objective
Before any hardware is dropped from any aircraft, the digital twin must be
able to (a) compare the three civil drop methods — free drop with airbag
pouch, guided round parachute, hang-line winch — under gusty wind, and
(b) **enforce the humanitarian guard**: never perform a ballistic release
inside the people-exclusion ring.

## 2. What was built (`src/airlab/guardian/drop.py`)
- `DropConfig` — fully declared constants (canopy 0.5 m², Cd 1.3, steer
  authority 3.5 m/s, release noise σ 0.6 m, gust σ 1.0 m/s, envelope
  12 m/s, exclusion ring 5 m...).  Nothing measured.
- `impact_samples` — Monte-Carlo (400 samples) of simplified, *declared*
  physics for each method.  Known (mean) wind is aimed-off by a jettison
  computer residual 8% misprediction; only the gust part is stochastic,
  and the guided chute cancels it up to a physical authority limit.
- `evaluate_method` → CEP50 / p90 / drift / descent time.
- `plan_drop` → the lowest-CEP **feasible** verdict, with guards:
  wind over the ops envelope → all methods rejected; people inside
  the 5 m ring → only `winch` (which never leaves the cleared hover
  vertical) remains allowed.
- Exported via `airlab.guardian` (`plan_drop`, `evaluate_method`, ...).

## 3. Demo (`out/guardian/drop.csv`, seed 11, h=25 m)
```
scenario                method            cep_m   feasible
h25_w2                  free_drop         1.11    1
h25_w2                  guided_parachute  0.92    1
h25_w2                  winch             0.98    1
h25_w6                  free_drop         1.13    1   (chosen: guided_parachute cep=0.94)
h25_w10                 free_drop         1.42    1   (winch cep=2.44 — swing grows with wind)
people_at_3m            winch             1.71    1   (only winch allowed in the ring)
wind_15_over_envelope   guided_parachute  1.19    0   wind 15.0 > envelope 12.0 m/s
```

Honest observations the twin produced (these are *sim-ahead* insights, not
flight claims): with aim-off, mild-wind CEPs are far below the design-doc
targets (≤5 m chute / ≤2 m winch), which sets the *measured* bar for the
hardware test campaign; and the winch degrades fastest with wind (line
swing), so gusty days favour the steerable chute at 25 m.

## 4. Tests (`TestPrecisionDrop` ×4)
guided keeps CEP/p90 < 5 m and ≤ free-drop+0.5; winch most precise but slow;
people at 3 m → winch-only; 15 m/s wind → everything rejected with the
envelope reason.  Full suite below.

## 5. Honest limits
- Simplified engineering simulator: declared constants, hover-based
  release, no canopy transient, no payload tumbling model.  It ranks
  methods and rejects unsafe states — it does **not** certify accuracy.
- Not connected yet to the browser flight sim (that is the natural next
  integration: press-to-drop with the same verdict UI).

## 6. Next autonomous steps
1. **Dynamic link-aware rebalancing** (program row) — keep safety loop,
   shed analytics when the mesh degrades.
2. **Real GCS integration** (MAVLink telemetry + supervisor board).
3. Wire `plan_drop` into the fly-demo so a student pilot must pass the
   same exclusion-ring guard as the design doc describes.

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py          # writes out/guardian/drop.csv
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestPrecisionDrop -v
```
