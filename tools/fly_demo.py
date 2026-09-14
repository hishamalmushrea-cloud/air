#!/usr/bin/env python3
"""Live flight-sim guardian server (safety research, simulated).

This is the *real* guardian code attached to a browser flight simulator:
the browser flies the quad (arcade physics, 60 fps) and streams its power
usage (throttle, AI compute load, airborne state) 5x per second.  This
server steps the true `PartThermalModel` from the repo with that power,
tracks a tiny battery-energy budget, and applies the real thermal-aware
trajectory logic:

  * `PredictiveRePlanner` (priority #12) decides whether a mission route
    is feasible at the nominal profile, feasible after a compute/power
    mitigation, or honestly rejected -- with the SAME code as
    `run_guardian.py`.
  * A live *throttle cap* protects an over-heating node during free flight:
    the nearer you get to a node limit, the more the guardian limits your
    thrust (and tells you to land).  No magic: less power, slower flight.

Everything is simulated with declared constants -- nothing is a measured
flight value.  Defensive/civil safety research only.

Run (repo root):
    PYTHONPATH=src .venv/bin/python tools/fly_demo.py --port 8081
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from airlab.guardian import PredictiveRePlanner          # noqa: E402
from airlab.guardian.thermal import PartThermalModel     # noqa: E402

# ------------------------- declared sim constants (tunable, not measured)
CRUISE_SPEED = 3.0          # m/s  nominal cruise
HOVER_POWER_W = 112.0       # W    propulsive power at full hover throttle
BATTERY_WH = 71.0           # Wh   energy budget
NOMINAL_CF = 0.30           # edge compute fraction at rest
HTML_PATH = os.path.join(HERE, "fly_demo.html")


class LiveVehicle:
    """One shared live vehicle state (single-pilot demo)."""

    def __init__(self, ambient_c: float = 25.0):
        self.ambient = float(ambient_c)
        self.lock = threading.Lock()
        self.reset()

    def reset(self, ambient: float = None):
        if ambient is not None:
            self.ambient = min(60.0, max(-20.0, float(ambient)))
        self.model = PartThermalModel(ambient_c=self.ambient)
        self.energy_wh_left = BATTERY_WH
        self.last = None
        self.throttle = 0.0
        self.compute_frac = NOMINAL_CF
        self.airborne = False

    def _power_w(self, throttle: float) -> float:
        # propulsive power ~ throttle^1.5 (declared prop curve), parked = 0.
        t = min(1.2, max(0.0, throttle))
        if not self.airborne:
            t = 0.0
        return HOVER_POWER_W * (t ** 1.5)

    def step(self, throttle: float, compute_frac: float,
             airborne: bool) -> dict:
        """Advance the live thermal model by the real elapsed wall-clock."""
        now = time.monotonic()
        with self.lock:
            dt = 0.0 if self.last is None else min(1.0, now - self.last)
            self.last = now
            self.throttle = throttle
            self.compute_frac = min(1.0, max(0.0, compute_frac))
            self.airborne = bool(airborne)
            power = self._power_w(throttle)
            if dt > 0.0:
                # sub-step for numerical stability
                n = max(1, int(dt / 0.1) + 1)
                sdt = dt / n
                for _ in range(n):
                    self.model.step(power, sdt, compute_frac=self.compute_frac)
                self.energy_wh_left = max(
                    0.0, self.energy_wh_left - power * dt / 3600.0
                    - 5.0 * self.compute_frac * dt / 3600.0)  # edge ~5 W
            temps = self.model.temperatures()
            limits = {n.name: n.max_temp_c for n in self.model.nodes}
            margins = {n.name: n.margin_c for n in self.model.nodes}
            worst = self.model.worst_node()
            worst_node = next((n for n in self.model.nodes if n.name == worst),
                              None)
            margin = worst_node.margin_c if worst_node is not None else 99.0
            over = worst_node.temp_c >= worst_node.max_temp_c \
                if worst_node is not None else False
            # Guardian intervention: soft cap as we approach the limit.
            if over:
                cap, alert = 0.30, "OVERHEAT - LAND NOW"
            elif margin < 2.0:
                cap, alert = round(0.45 + 0.10 * margin, 2), "HOT - REDUCING"
            elif margin < 6.0:
                cap, alert = round(0.80 + 0.03 * (margin - 2.0), 2), "WARM"
            else:
                cap, alert = 1.0, "NOMINAL"
            soc = self.energy_wh_left / BATTERY_WH
            if soc < 0.10 and not over:
                alert = "BATTERY LOW - LAND"
            return {
                "temps": {k: round(v, 2) for k, v in temps.items()},
                "limits": limits,
                "margins": {k: round(v, 2) for k, v in margins.items()},
                "worst_node": worst,
                "worst_c": round(max(temps.values()), 2),
                "margin_c": round(margin, 2),
                "throttle_cap": cap,
                "alert": alert,
                "power_w": round(power + 5.0 * self.compute_frac, 1),
                "ambient_c": self.ambient,
                "battery_frac": round(soc, 3),
                "initial_temps": {k: round(v, 2) for k, v in temps.items()},
            }

    def plan_mission(self, route_len: float, battery_frac: float) -> dict:
        """Real thermal-aware trajectory check using the LIVE part temps."""
        with self.lock:
            init = {n.name: n.temp_c for n in self.model.nodes}
        planner = PredictiveRePlanner(
            thermal_aware=True, thermal_ambient_c=self.ambient,
            thermal_initial_temps=init,
            cruise_speed=CRUISE_SPEED, hover_power_w=HOVER_POWER_W,
            battery_capacity_wh=BATTERY_WH,
            lateral_offsets=(0.0,), vertical_offsets=(0.0,))
        start = np.array([0.0, 0.0, -2.0])
        goals = [np.array([float(route_len), 0.0, -2.0])]
        res = planner.plan(start, goals, battery_frac=battery_frac)
        return {
            "decision": ("mitigated" if res.thermal_mitigated
                         else ("ok" if res.thermal_feasible else "rejected")),
            "feasible": bool(res.feasible),
            "thermal_feasible": bool(res.thermal_feasible),
            "mitigated": bool(res.thermal_mitigated),
            "compute_frac": res.thermal_compute_frac,
            "power_frac": res.thermal_power_frac,
            "power_w": round(res.thermal_power_w, 1),
            "worst_node": res.thermal_worst_node,
            "max_temp_c": round(float(res.thermal_max_c), 2),
            "margin_c": None if abs(res.thermal_margin_c) == float("inf")
            else round(float(res.thermal_margin_c), 2),
            "reasons": list(res.reasons),
            "route_len_m": float(route_len),
            "mission_time_s": round(res.repl_length /
                                    (CRUISE_SPEED * max(0.35, res.thermal_power_frac)), 1),
            "initial_temps": {k: round(v, 1) for k, v in init.items()},
        }


VEHICLE = LiveVehicle()


def _clamp(v, lo, hi):
    return float(min(hi, max(lo, v)))


def _num(q, name, default, lo, hi):
    try:
        return _clamp(float(q.get(name, [default])[0]), lo, hi)
    except (TypeError, ValueError):
        return float(default)


class Handler(BaseHTTPRequestHandler):
    server_version = "FlyDemo/1.0"

    def _send(self, code: int, body: bytes,
              ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _read_json(self) -> dict:
        try:
            ln = int(self.headers.get("Content-Length", "0"))
            if ln <= 0 or ln > 16384:
                return {}
            return json.loads(self.rfile.read(ln).decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif url.path == "/healthz":
                self._send(200, b"ok", "text/plain")
            elif url.path == "/api/state":
                q = parse_qs(url.query)
                self._json(VEHICLE.step(
                    throttle=_num(q, "throttle", 0.0, 0.0, 1.2),
                    compute_frac=_num(q, "cf", NOMINAL_CF, 0.0, 1.0),
                    airborne=bool(int(_num(q, "airborne", 1, 0, 1)))))
            elif url.path == "/api/mission":
                q = parse_qs(url.query)
                self._json(VEHICLE.plan_mission(
                    route_len=_num(q, "route_len", 60.0, 5.0, 400.0),
                    battery_frac=_num(q, "battery_frac", 1.0, 0.0, 1.0)))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": "internal error", "detail": str(exc)}, 500)

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/api/state":
                body = self._read_json()
                self._json(VEHICLE.step(
                    throttle=_clamp(float(body.get("throttle", 0.0)), 0.0, 1.2),
                    compute_frac=_clamp(float(body.get("cf", NOMINAL_CF)),
                                        0.0, 1.0),
                    airborne=bool(body.get("airborne", True))))
            elif url.path == "/api/reset":
                body = self._read_json()
                ambient = body.get("ambient")
                VEHICLE.reset(None if ambient is None else
                              _clamp(float(ambient), -20.0, 60.0))
                self._json({"ok": True, "ambient_c": VEHICLE.ambient})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": "internal error", "detail": str(exc)}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("[fly] %s\n" % (fmt % args))


def main() -> int:
    ap = argparse.ArgumentParser(description="Live guardian flight sim "
                                             "(simulated, defensive).")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"live guardian flight sim on http://{args.host}:{args.port} "
          f"(simulated, defensive research)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
