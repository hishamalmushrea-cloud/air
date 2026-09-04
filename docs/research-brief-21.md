# Research Brief #21 — Persistence-Gated Flow-Source Rejection is a Tradeoff, Not a Win

**Date:** 2026-09-03
**Project:** AIR Lab
**Status:** implemented, tested.  The persistence gate is now a tunable knob,
kept **off by default** (immediate rejection, the characterised safe point for
persistent faults).  The brief records why it is not promoted.

---

## 1. Goal

Brief #20 showed that immediate flow-source rejection on a **transient** warn can
turn a self-healing event into a crash (the landmark-only arm crashed 2/6).  The
natural fix is a **persistence gate**: only reject the velocity-aiding source
after the independent detector consensus has been below `warn` for a sustained
window, so a ~0.2 s transient dip does not strip a source that will recover.

## 2. Implementation

- `SafetyConfig.flow_reject_persist_s` (default **0.0** = original immediate
  rejection).
- `SimConfig.flow_reject_persist_s` (optional override) and
  `Scenario.flow_reject_persist_s`.
- `consensus_study(..., flow_reject_persist_s=...)` and
  `--flow-reject-persist <seconds>` CLI.
- Rejection now accumulates `_det_low_accum` while `det_health < warn` and only
  fires when it reaches the gate; it resets on recovery.

Why default 0.0: see the results.  The gate is exposed as a tuning knob, not made
the default.

## 3. Mechanism (instrumented, scen0, transient)

The detector dip under the transient is **very brief**:

```
warn 0.55
first below t 9.8   last below t 9.99
longest sustained below streak ~0.2 s
min det 0.528
```

So a 0.5 s gate lets that warn recover; the same gate barely changes a genuine
persistent fault (which stays low for many seconds).

## 4. Results

### 4a. Transient fault hidden inside sparse-FG (n=6, seed 909, 8–18 s inside 5–28 s)

| policy | gate=0.0 (immediate) | gate=0.5 (persist) |
|---|---|---|
| none | 0.000 / 0 landed | 0.000 / 0 landed |
| lm_only | **0.097**, 2/6 crash | **0.010**, 1/6 crash |
| fg_only | 0.005 / 0 landed | 0.007 / 0 landed |
| adaptive | 0.016, 0 crash | **0.060, 1/6 crash** |
| adaptive_veto_trust | 0.000, 0 crash | 0.000, 0 crash |

Reading: the gate helps `lm_only` (2/6 → 1/6) but **worsens `adaptive`** (0 crash
→ 1/6 crash) on the same grid.  `adaptive_veto_trust` is unchanged (0 crash).

### 4b. Persistent bias.25 + sparse-FG (n=6, seed 909, gate=0.5)

| policy | in_bounds | landed | crash time-frac | crash events |
|---|---|---|---|---|
| lm_only | 0.725 | 0.161 | **0.089** | 1/6 |
| adaptive | 0.849 | 0.816 | 0.000 | 0/6 |
| adaptive_veto_trust | 0.849 | 0.816 | 0.000 | 0/6 |

Compared with immediate rejection (brief #18, lm_only bias.25 + sparse FG:
landed 0.170, crash 0 at n=3), the gate **under-protects a persistent fault on
the landmark-only arm** — delaying rejection lets a persistent corrupt source
drive longer.

## 5. Honest reading

The persistence gate is exactly the kind of design "fix" that looks right in the
single case it targets and has an opposite effect in the case it was meant not to
touch:

- **Transient fault:** gate reduces false-rejection crashes on `lm_only`
  (the arm that is most exposed because it has no FG to veto).
- **Persistent fault:** gate delays rejection and can *increase* crashes on
  `lm_only` (0 → 0.089 time-frac / 1/6).
- **`adaptive_veto_trust`** is already robust on both because it is selectively
  conservative about credibility, so a generic gate change adds no benefit there
  and can add risk.
- **`adaptive`** got slightly worse on the transient grid with the gate (1/6).

Conclusion: a fixed persistence gate is **not** a strict safety improvement; it
trades one failure mode for another.  We keep immediate rejection as the default
(the characterised point) and expose the gate only as a tunable knob for
operation-specific tuning.

## 6. The right next idea (not done in this brief)

The correct design is **recovery-aware decision-making**, not a fixed time gate:
- Keep *detection* aware that a fault can be transient (the detector should
  report "fault + recoverable vs persistent", not just "health low").
- Only reject the source when the independent consensus is *still* low *and* the
  vehicle is heading outside bounds (the cost of not rejecting is concrete).
- Optionally **re-enable** a rejected source if it demonstrably recovers and GPS
  is available, so a single transient warn does not permanently strip a source.

This is a bigger change and should be its own experiment.

## 7. Tests

- `test_flow_reject_persist_gate_is_a_tradeoff_not_a_win`: scen0 transient —
  immediate (0.0) rejects too early and crashes; gate 2.0 lets the ~0.2 s warn
  recover (crash 0).
- Existing `test_transient_in_sparse_fg_does_not_crash_under_trust`: still passes
  with the default (0.0): `lm_only` crashes, `adaptive_veto_trust` does not.
- Full suite: **39/39 OK** (after adding the gate test).

## 8. Run

```bash
# immediate (default)
.venv/bin/python -u run_consensus.py --n 6 --duration 40 --seed 909 \
  --fg-flow-outage 5-28 --faults transient.3 \
  --policies none,lm_only,fg_only,adaptive,adaptive_veto_trust \
  --adaptive-escalate 0.65 --out out/consensus_transient_fgsparse_n6.csv

# persistence gate 0.5 s
.venv/bin/python -u run_consensus.py --n 6 --duration 40 --seed 909 \
  --fg-flow-outage 5-28 --faults transient.3 \
  --policies none,lm_only,fg_only,adaptive,adaptive_veto_trust \
  --adaptive-escalate 0.65 --flow-reject-persist 0.5 \
  --out out/consensus_transient_persist_n6.csv
```
