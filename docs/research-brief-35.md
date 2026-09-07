# Research Brief #35 — Edge-vs-Ground Compute Split

**Date:** 2026-09-05
**Project:** AIR Lab (Nexus-Predator)
**Status:** implemented, tested.  Program priority #10.  Directly implements
master prompt section 18.

---

## 1. Objective
Section 18 asks: which compute must run **onboard**, which at the **ground
station**, and which at the **server/cloud** — compared over latency, power,
reliability, bandwidth, privacy, and compute requirements.  This brief answers
that for the guardian's compute tasks transparently.

## 2. What was built

### `airlab.guardian.edge_split`
- `ComputeTask` — a task with declared-estimate `topps`, `power_w`,
  `latency_ms`, `safety_critical`, `data_mbps`, `privacy_sensitive`.
- `LinkEstimate` — declared ground/cloud RTT, link bandwidth, reliability.
- `EdgeGroundSplit.place(tasks)` — classifies each task and renders a
  `PlacementResult` + `SplitResult`.
- `SplitResult.summary()` / `score` — per-task placement + aggregate edge load.

**Placement rules (hard safety rule):**
- **Safety-critical** tasks (threat detection, guardian evasion, predictive
  re-plan, sensor fusion) **must stay onboard** — a loss of link can never
  remove the aircraft's ability to protect itself.
- Privacy-sensitive + low-bandwidth analytics stay onboard (data cannot leave
  a 1 Mbps link).
- High-latency, non-private analytics → **cloud**; moderate-latency → **GCS**;
  otherwise onboard.

## 3. Demo (`out/guardian/edge_split.csv`, 20 Mbps link)
```
safety tasks all onboard=True
onboard_topps=0.210   onboard_power=4.9 W
edge_power_frac=0.613   edge_topps_frac=0.600
offload_frac=0.84   data=10.5 Mbps   ground_latency=1200.0 ms
```

The allocator keeps all 4 safety tasks onboard and offloads ~84 % of the
analytics data (10.5 Mbps) to GCS/cloud, while the 0.35 TOPS edge stays inside
~61 % of its 8 W power budget and ~60 % of its TOPS budget.

## 4. Tests (`TestEdgeSplit`, 4 tests)
- all 4 safety-critical tasks are placed onboard with the safety reason.
- latency-tolerant analytics move to ground/cloud under a high-bandwidth link.
- privacy-sensitive + low-bandwidth analytics stay onboard.
- aggregate reports edge TOPS/power budget and `safety_full_onboard=True`.
- Guardian suite: **47/47**; full suite below.

## 5. Honest limits
- No real GCS/cloud service, no real network.  `LinkEstimate` values are
  **declared estimates**.
- Compute/power/latency per task are declared estimates, not measurements on
  the real edge board.
- The split is static per task class; it does not yet do dynamic
  degradation-aware rebalancing (reducing analytics when the link drops).
- Privacy is a two-level flag only; no PII/data-governance model yet.

## 6. Next autonomous tasks
1. **Thermal-aware trajectory** (modulate `compute_frac`/cruise power so a hot
   route can cool, not just reject).
2. **Dynamic link-aware rebalancing** (drop/queue analytics when link degrades,
   keep the safety loop untouched).
3. **Real GCS integration** (MAVLink telemetry + a supervisor board).

## 7. Run
```bash
PYTHONPATH=src .venv/bin/python run_guardian.py
PYTHONPATH=src .venv/bin/python -m unittest tests.test_guardian.TestEdgeSplit -v
```
