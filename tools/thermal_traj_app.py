#!/usr/bin/env python3
"""Interactive thermal-aware trajectory demo (safety research, simulated).

Serves a small browser dashboard where you set the *live* part temperatures,
ambient, route length and battery level, then asks the real
`PredictiveRePlanner` whether the route is:
  * feasible at the nominal profile (no mitigation),
  * feasible after thermal mitigation (compute_frac / power_frac reduced),
  * rejected (no safe in-range profile).

Everything shown is modelled/declared and simulated -- nothing here is a
measured flight value.  Defensive/safety research only.

Run (from the repo root):
    PYTHONPATH=src .venv/bin/python tools/thermal_traj_app.py --port 8080

The server needs only the Python stdlib + numpy already required by the repo.
Binds 0.0.0.0 so the Arena live preview (and a normal browser) can reach it.
"""

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.guardian import PredictiveRePlanner          # noqa: E402
from airlab.guardian.thermal import PartThermalModel     # noqa: E402

CRUISE_SPEED = 3.0
HOVER_POWER_W = 112.0
BATTERY_WH = 71.0
NOMINAL_CF = 0.3
HTML_PATH = os.path.join(HERE, "thermal_traj_app.html")

_lock = threading.Lock()
_sweep_cache: dict = {}  # key -> payload (bounded below)


# ---------------------------------------------------------------- helpers
def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _read_params(query: dict) -> dict:
    """Sanitised scenario parameters (never trust the client)."""
    def num(name, default, lo, hi):
        try:
            return _clamp(float(query.get(name, [default])[0]), lo, hi)
        except (TypeError, ValueError):
            return float(default)

    return {
        "cpu": num("cpu", 59.0, 10.0, 120.0),
        "esc": num("esc", 30.0, 10.0, 120.0),
        "motor": num("motor", 30.0, 10.0, 120.0),
        "battery": num("battery", 30.0, 10.0, 120.0),
        "ambient": num("ambient", 25.0, -20.0, 60.0),
        "route_len": num("route_len", 12.0, 5.0, 400.0),
        "battery_frac": num("battery_frac", 1.0, 0.0, 1.0),
    }


def _plan(p: dict):
    """Run the real planner on a straight route and return the decision."""
    init = {"cpu_npu": p["cpu"], "esc": p["esc"],
            "motor": p["motor"], "battery": p["battery"]}
    planner = PredictiveRePlanner(
        thermal_aware=True, thermal_ambient_c=p["ambient"],
        thermal_initial_temps=init,
        cruise_speed=CRUISE_SPEED, hover_power_w=HOVER_POWER_W,
        battery_capacity_wh=BATTERY_WH,
        lateral_offsets=(0.0,), vertical_offsets=(0.0,))
    start = np.array([0.0, 0.0, -2.0])
    goals = [np.array([p["route_len"], 0.0, -2.0])]
    return planner.plan(start, goals, battery_frac=p["battery_frac"])


def _timeline(p: dict, compute_frac: float, power_frac: float,
              max_points: int = 200) -> dict:
    """Per-node temperature history for one profile (sim for demo)."""
    init = {"cpu_npu": p["cpu"], "esc": p["esc"],
            "motor": p["motor"], "battery": p["battery"]}
    model = PartThermalModel(ambient_c=p["ambient"], initial_temps=init)
    speed = max(0.2, CRUISE_SPEED * _clamp(power_frac, 0.1, 1.0))
    t_total = max(0.1, p["route_len"] / speed)
    steps = max(1, int(min(t_total, 7200.0) / 0.5))
    dt = t_total / steps
    every = max(1, steps // max_points)
    t_axis, hist = [0.0], {n.name: [round(t, 2)]
                           for n, t in zip(model.nodes, model.temperatures().values())}
    for i in range(1, steps + 1):
        model.step(HOVER_POWER_W * power_frac, dt,
                   compute_frac=_clamp(compute_frac, 0.0, 1.0))
        if i % every == 0 or i == steps:
            t_axis.append(round(i * dt, 1))
            for name, t in model.temperatures().items():
                hist[name].append(round(t, 2))
    return {"t": t_axis, "nodes": hist, "t_total_s": round(t_total, 1),
            "speed_mps": round(speed, 2), "power_w": round(HOVER_POWER_W * power_frac, 1)}


def _limits() -> dict:
    m = PartThermalModel()
    return {n.name: n.max_temp_c for n in m.nodes}


def handle_plan(query: dict) -> dict:
    p = _read_params(query)
    res = _plan(p)
    nominal = _timeline(p, NOMINAL_CF, 1.0)
    chosen = nominal if not res.thermal_mitigated else _timeline(
        p, res.thermal_compute_frac, res.thermal_power_frac)
    return {
        "params": p,
        "decision": ("mitigated" if res.thermal_mitigated
                     else ("ok" if res.thermal_feasible else "rejected")),
        "feasible": bool(res.feasible),
        "thermal_feasible": bool(res.thermal_feasible),
        "mitigated": bool(res.thermal_mitigated),
        "compute_frac": res.thermal_compute_frac,
        "power_frac": res.thermal_power_frac,
        "power_w": res.thermal_power_w,
        "worst_node": res.thermal_worst_node,
        "max_temp_c": round(float(res.thermal_max_c), 2),
        "margin_c": None if res.thermal_margin_c in (float("inf"), float("-inf"))
        else round(float(res.thermal_margin_c), 2),
        "reasons": list(res.reasons),
        "limits": _limits(),
        "nominal_profile_w": HOVER_POWER_W,
        "nominal_compute_frac": NOMINAL_CF,
        "timeline_nominal": nominal,
        "timeline_chosen": chosen,
    }


def handle_sweep(query: dict) -> dict:
    p = _read_params(query)
    lo, hi = p["ambient"] + 15.0, max(p["limits_cpu"] if "limits_cpu" in p else 75.0,
                                      p["ambient"] + 16.0)
    key = (round(p["esc"], 1), round(p["motor"], 1), round(p["battery"], 1),
           round(p["ambient"], 1), round(p["route_len"], 1),
           round(p["battery_frac"], 2))
    with _lock:
        if key in _sweep_cache:
            return _sweep_cache[key]
    points = []
    cpu = round(lo, 1)
    while cpu <= hi + 1e-9 and len(points) < 40:
        q = dict(p, cpu=cpu)
        res = _plan(q)
        points.append({
            "cpu": round(cpu, 1),
            "decision": ("mitigated" if res.thermal_mitigated
                         else ("ok" if res.thermal_feasible else "rejected")),
            "power_frac": res.thermal_power_frac,
            "max_temp_c": round(float(res.thermal_max_c), 2),
        })
        cpu += 1.0
    payload = {"cpu_from": lo, "cpu_to": hi, "points": points,
               "params": p, "limits": _limits()}
    with _lock:
        if len(_sweep_cache) > 200:
            _sweep_cache.clear()
        _sweep_cache[key] = payload
    return payload


# ---------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "ThermalTrajDemo/1.0"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def do_GET(self):  # noqa: N802 (stdlib API)
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/healthz":
                self._send(200, b"ok", "text/plain")
            elif url.path == "/api/plan":
                self._json(handle_plan(parse_qs(url.query)))
            elif url.path == "/api/sweep":
                self._json(handle_sweep(parse_qs(url.query)))
            elif url.path == "/api/limits":
                self._json({"limits": _limits(),
                            "nominal_profile_w": HOVER_POWER_W,
                            "nominal_compute_frac": NOMINAL_CF})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # never leak a traceback to the client
            self._json({"error": "internal error", "detail": str(exc)}, 500)

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[demo] %s\n" % (fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description="Interactive thermal-aware "
                                             "trajectory demo (simulated).")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"thermal-trajectory demo on http://{args.host}:{args.port} "
          f"(simulated, defensive research)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
