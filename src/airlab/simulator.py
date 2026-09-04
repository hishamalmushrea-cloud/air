"""Orchestrator: connect vehicle + sensors + estimator + controller + mission.

Run at a base simulation step of 10 ms. Each subsystem samples at its own rate
(IMU 100 Hz, AHRS 50 Hz, baro 20 Hz, GPS 10 Hz), exactly like a real flight
stack, so that the fusion has to interpolate and *fuse* rather than being
handed perfectly co-located measurements.
"""

from __future__ import annotations

import numpy as np

from .dynamics import Quadrotor
from .sensors import SensorSuite, SensorConfig
from .ahrs import ComplementaryAHRS
from .fusion import (NavEKF, GPS, BARO, AHRS, FLOW,
                     S_IDX, V_IDX, R_IDX, BG_IDX, BA_IDX, _wrap_state)
from .control import FlightController
from .mission import WaypointMission
from .metrics import evaluate_mission
from .safety import (SafetyMonitor, SafetyConfig, CRUISE, HOLD, LAND, LANDED,
                     RTL, VelocityIntegrityMonitor)
from .energy import PowerModel
from .landmarks import LandmarkField, LandmarkConsistency
from .factorgraph import SlidingFactorGraph, build_keyframe, ImuPreintegrator
from .trust import FrameTrustLearner


def _cone_bearing(axis: np.ndarray, a_rad: float, b_rad: float) -> np.ndarray:
    """A unit vector near ``axis`` (within ~cone half-angle).

    Builds an orthonormal basis around ``axis`` and perturbs by two small
    orthogonal angles, emulating a camera whose features all lie in a small
    projected patch.
    """
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    axis = axis / n
    u = np.array([1.0, 0.0, 0.0])
    if abs(axis[0]) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    u = u - axis * np.dot(axis, u)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    d = axis + a_rad * u + b_rad * v
    return d / np.linalg.norm(d)


class SimConfig:
    def __init__(self) -> None:
        self.dt = 0.01
        self.duration = 40.0
        self.wind_ned = np.array([0.0, 0.0, 0.0])
        # Fault scenario: GNSS outage window (seconds) or None.
        self.gps_outage = None  # (start, end)
        self.gps_outage_scale = 1.0
        self.seed = 7

        # Gate optional velocity-aiding sensor (optical flow / VIO surrogate).
        self.flow_enabled = True
        # Velocity-aiding faults (windows / scales / bias ramp).
        self.flow_outage = None              # (start, end)
        self.flow_scale = 1.0
        self.flow_bias_ramp = 0.0
        # Sparse factor-graph window: the graph stops receiving flow factors
        # (under-determined) but the EKF keeps its flow aiding.  This isolates
        # the independent FG detector's weak-voice / under-determined behaviour
        # from the velocity-aiding itself.  See research-brief-18.
        self.factorgraph_flow_outage = None  # (start, end)

        # Uncertainty-aware safety layer.
        self.safety_enabled = True
        self.safety_kwargs = {}
        # Optional parallel IMU dead-reckon / integrity check.  Currently off
        # by default: an open-loop IMU dead-reckon is not reliable enough as an
        # independent monitor over long outages (see research-brief-02).
        self.integrity_monitor_enabled = False

        # Landmark-based independent consistency check.  This is the preferred
        # "second opinion": it observes true world geometry, so it can catch a
        # corrupt velocity-aiding source that the EKF would otherwise trust.
        self.landmark_enabled = True
        self.landmark_hz = 5.0
        # Window during which the camera reports no/new landmarks (feature-poor
        # flight).  The landmark score holds its last opinion, but the
        # availability weight drops toward 0 — this is how we exercise the
        # measurement-availability guardrail.
        self.landmark_outage = None   # (start, end) seconds

        # Window during which the camera still reports *many* landmarks but all
        # within a tiny angular cone (a degenerate low-parallax scene, e.g. a
        # close wall).  Raw count is high, so the binary availability weight
        # cannot see it; the angular-diversity trust can and should treat it as
        # thin.  This is the fault mode that differentiates adaptive_veto_trust.
        self.landmark_cluster = None  # (start, end) seconds
        self.landmark_cluster_cone = 0.04  # rad: half-angle of the degenerate cone

        # Self-calibrated per-frame trust: collect the frame-informativeness
        # distribution on a healthy startup, then score every later frame
        # against it.  Replaces hand-set count/3 and rms/1.2 constants.
        self.trust_learn = True
        self.trust_calibrate_s = 6.0   # healthy startup window (s)

        # Principled factor-graph consistency monitor (IMU+flow+GPS+landmarks,
        # jointly optimised with a robust kernel).  Now enabled by default as a
        # second, structurally independent safety opinion alongside the
        # landmark detector.  Its health feeds a dedicated safety path that can
        # force a landing even while GPS is still available, which closes the
        # gap where a corrupt velocity source drove the aircraft into the
        # ground during a GNSS-aided phase.  See research-brief-06.
        self.factorgraph_enabled = True
        self.factorgraph_hz = 2.0
        self.factorgraph_kwargs = {}

        # How to combine the independent detectors into a single
        # ``detector_health`` for the safety layer:
        #   "min"     -> worst-of (OR): any bad detector triggers  (most conservative)
        #   "max"     -> best-of (AND): both must agree a fault exists (least alarm)
        #   "geom"    -> geometric mean (soft consensus)
        #   "weighted"/"adaptive_weighted" -> availability-weighted soft
        #                consensus: a detector with little local data (few
        #                landmarks, under-determined graph) is down-weighted,
        #                so it cannot force a decision on thin evidence.
        #   "adaptive_veto" -> symmetric-availability guardrail: a detector
        #                with thin data may NOT *trigger* escalation, but it
        #                KEEPS its full healthy voice to veto one (this avoids
        #                a stale-healthy landmark being down-weighted and
        #                letting a noisier factor graph false-land a healthy
        #                mission during a camera outage).
        # ``adaptive`` is the default: it keeps `min`-level crash protection on
        # deep faults (0.841 vs 0.845 landed at bias.25) while cutting the
        # benign-fault mission cost by ~33% (0.142 vs 0.212).  See
        # research-brief-10.
        self.detector_consensus = "adaptive"
        self.adaptive_escalate = 0.65   # soft->worst-of escalation line

        # Mission-aware emergency response: instead of always landing where the
        # fault is detected, try to return to base first if the battery and
        # geometry safely allow it; otherwise land immediately.
        self.mission_aware = False
        self.base_pos = (0.0, 0.0)          # NED north/east of takeoff
        self.rtl_speed = 2.0                 # m/s cruise back to base
        self.rtl_radius_m = 2.0              # "at base" tolerance
        self.energy_reserve_frac = 0.25      # need >= this to attempt RTL
        self.battery_capacity_wh = 100.0

        self.vehicle_kwargs = {}
        self.sensor_kwargs = {}
        self.ekf_kwargs = {"r_scale": 1.0, "q_scale": 1.0}
        self.controller_kwargs = {}
        # Ablation flag: feed the controller perfect state (for debugging /
        # separating control performance from estimation performance).
        self.truth_estimate = False

        # Waypoints are (north, east, height) with height positive-up.
        self.waypoints = [
            (0.0, 0.0, 2.0),
            (8.0, 0.0, 3.0),
            (8.0, 8.0, 4.0),
            (-2.0, 8.0, 5.0),
            (-2.0, -2.0, 5.0),
        ]
        self.cruise_speed = 2.0


class SimRun:
    """A single simulation run; keeps every recorded signal."""

    def __init__(self) -> None:
        self.t = []
        self.true_pos = []
        self.true_vel = []
        self.true_rpy = []
        self.est_pos = []
        self.est_vel = []
        self.est_rpy = []
        self.ref_pos = []
        self.ref_vel = []
        self.ref_yaw = []
        self.gps_available = []
        self.imu = []
        self.control = []
        self.cost = []
        self.mode = []
        self.reason = []
        self.unc_horiz_std = []
        self.flow_health = []
        self.flow_mismatch = []
        self.flow_dr_vel = []
        self.landmark_score = []
        self.landmark_residual = []
        self.factorgraph_health = []
        self.factorgraph_residual = []
        self.landed = []

    def record(self, sim, pos_ref, vel_ref, yaw_ref) -> None:
        self.t.append(sim.time)
        self.true_pos.append(sim.vehicle.pos.copy())
        self.true_vel.append(sim.vehicle.vel.copy())
        self.true_rpy.append(sim.vehicle.rpy.copy())
        self.est_pos.append(sim.ekf.pos)
        self.est_vel.append(sim.ekf.vel)
        self.est_rpy.append(sim.ekf.rpy)
        self.ref_pos.append(pos_ref)
        self.ref_vel.append(vel_ref)
        self.ref_yaw.append(yaw_ref)
        self.gps_available.append(1.0 if sim.sensors.cfg.gps_dropout <= 0.0 else 0.0)
        self.imu.append([sim.last_imu.accel.copy(), sim.last_imu.gyro.copy()])
        self.control.append(sim.last_control.copy())
        self.cost.append(float(np.linalg.norm(sim.vehicle.pos - sim.ekf.pos)))
        self.mode.append(sim.safety.mode)
        self.reason.append(sim.safety.reason)
        self.unc_horiz_std.append(sim._last_unc_std)
        self.flow_health.append(sim._effective_flow_health)
        self.flow_mismatch.append(sim._last_flow_mismatch)
        self.flow_dr_vel.append(sim.flow_integrity.vel.copy())
        self.landmark_score.append(sim._landmark_score)
        self.landmark_residual.append(sim._landmark_residual)
        self.factorgraph_health.append(sim._factorgraph_health)
        self.factorgraph_residual.append(sim._factorgraph_residual)
        self.landed.append(1.0 if sim.safety.mode == LANDED else 0.0)

    def as_arrays(self) -> dict:
        d = dict(self.__dict__)
        for k in ("t", "true_pos", "true_vel", "true_rpy", "est_pos", "est_vel",
                  "est_rpy", "ref_pos", "ref_vel", "control", "cost",
                  "unc_horiz_std", "flow_health", "flow_mismatch",
                  "flow_dr_vel", "landmark_score", "landmark_residual",
                  "factorgraph_health", "factorgraph_residual", "landed"):
            d[k] = np.asarray(d[k], dtype=float)
        d["ref_yaw"] = np.asarray(d["ref_yaw"], dtype=float)
        d["gps_available"] = np.asarray(d["gps_available"], dtype=float)
        d["mode"] = d["mode"]
        d["reason"] = d["reason"]
        d["imu"] = d["imu"]
        return d


class Simulator:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config if config is not None else SimConfig()
        self.vehicle = self._make_vehicle(self.cfg)
        self.sensors = self._make_sensors(self.cfg)
        self.ahrs = ComplementaryAHRS(rpy=self.vehicle.rpy)
        self.ekf = NavEKF(self.vehicle.pos, self.vehicle.vel, self.vehicle.rpy,
                          backend="numeric", **self.cfg.ekf_kwargs)
        self.controller = FlightController(**self.cfg.controller_kwargs)
        self.mission = WaypointMission(self.cfg.waypoints, speed=self.cfg.cruise_speed)
        self.safety = SafetyMonitor(SafetyConfig(**self.cfg.safety_kwargs))
        self.safety.cfg.enabled = self.cfg.safety_enabled
        self.safety.cfg.adaptive_escalate = float(self.cfg.adaptive_escalate)
        self.flow_integrity = VelocityIntegrityMonitor()
        self.flow_integrity.reset(self.vehicle.vel, self.vehicle.rpy)
        # IMU-only dead-reckon position, independent of the flow-fed EKF.
        self._dr_pos = self.vehicle.pos.copy()
        self._dr_vel = self.vehicle.vel.copy()
        # Proper IMU preintegrator (re-anchored to GPS and to the factor graph's
        # optimized last keyframe), so the graph is seeded from a clean relative
        # pose rather than a drifted absolute dead-reckon.
        self.imu_preint = ImuPreintegrator()
        self.imu_preint.reset(self.vehicle.pos, self.vehicle.vel)
        self.landmark_field = LandmarkField()
        self.landmark_consistency = LandmarkConsistency(self.landmark_field)
        self.factorgraph = SlidingFactorGraph(self.landmark_field.positions,
                                              **self.cfg.factorgraph_kwargs)
        # Online energy accounting for mission-aware emergency response.
        self.power_model = PowerModel(battery_capacity_wh=self.cfg.battery_capacity_wh)
        self._energy_wh_used = 0.0
        self.rng = np.random.default_rng(self.cfg.seed)
        self.time = 0.0
        self.last_imu = None
        self.last_control = np.zeros(4)
        self._last_unc_std = 0.0
        self._effective_flow_health = 1.0
        self._last_flow_mismatch = 0.0
        self._landmark_score = 1.0
        self._landmark_residual = 0.0
        self._landmark_obs_count = 0    # observed landmarks in latest frame
        self._landmark_dirs = np.zeros((0, 3))  # latest observed body dirs
        self._factorgraph_health = 1.0
        self._factorgraph_residual = 0.0
        self._fg_flow_components = 0     # flow factors used in last optimisation
        self._fg_converged = False
        # Startup calibration of the factor-graph healthy baseline.
        self._fg_calibrated = False
        self._fg_baseline = None
        self._fg_calibration_samples = []
        # The factor graph's health is not meaningful until its baseline is
        # calibrated from trusted GNSS data, so the safety layer must not use
        # it before that point.
        self.factorgraph_health_trusted = False
        # Self-calibrated per-frame informative-frame trust (landmark /
        # factor-graph).  Learns the healthy startup distribution instead of
        # using hand-set count/3 and rms/1.2 constants.
        self._lm_trust = FrameTrustLearner("landmark")
        self._fg_trust = FrameTrustLearner("factorgraph")
        # Detector-triggered source rejection: once the independent monitors
        # agree the velocity-aiding source is corrupt, stop feeding it to the
        # EKF.  This is what makes a later RTL safe — navigation continues on
        # GPS/baro only, and the corrupt velocity can no longer drive the
        # controller.
        self.flow_rejected = False
        self._flow_reject_reason = "none"

    def _detector_weight(self, kind: str) -> float:
        """Local measurement-availability weight for a detector.

        This is deliberately *not* derived from the detector's own verdict
        (a faulty detector must not down-weight itself and hide the fault).  It
        only measures whether the detector had enough local data to form an
        opinion:

          landmark : fraction of the min useful landmark count actually seen
                     (a camera in a feature-poor area is a weak voice)
          factorgraph : fraction of the graph's minimum factor count actually
                     available (a graph with too few factors is under-determined)
        """
        if kind == "landmark":
            return float(np.clip(self._landmark_obs_count / 3.0, 0.0, 1.0))
        if kind == "factorgraph":
            if not self.factorgraph_health_trusted:
                return 0.0
            comp = self._fg_flow_components
            base = float(np.clip(comp / max(self.factorgraph.min_keyframes, 1), 0.0, 1.0))
            conv = 1.0 if self._fg_converged else 0.5
            return float(base * conv)
        return 1.0

    def _combine_detectors(self, parts: list[float]) -> float:
        """Combine independent detector healths using the consensus policy.

        "min"   -> worst-of (OR): any bad detector triggers (most conservative)
        "max"   -> best-of (AND): both must agree a fault exists (least alarm)
        "geom"  -> geometric mean (soft consensus)
        #   "adaptive" -> soft consensus while it still vouches for the sensors;
        #                the moment even the soft consensus drops below the
        #                detector warn threshold, escalate to the worst-of
        #                opinion.  This keeps mild (survivable) faults from
        #                forcing a needless landing while still escalating
        #                decisively on deep (diverging) faults.
        "weighted" / adaptive soft uses a *measured-availability weighted*
        geometric mean: a detector with little local data (few landmarks, an
        under-determined graph) is down-weighted so it cannot force a decision
        on thin evidence, while a detector with good data keeps its full voice.
        """
        if not parts:
            return 1.0
        a = np.asarray(parts, dtype=float)
        mode = self.cfg.detector_consensus
        if mode == "max":            # AND: both must agree there is a fault
            return float(np.max(a))
        if mode in ("geom", "adaptive", "weighted", "adaptive_weighted",
                    "adaptive_veto", "adaptive_veto_trust"):
            w = self._consensus_weights(len(parts))
            soft_uw = float(np.sqrt(np.prod(np.clip(a, 1e-9, 1.0))))
            soft_w = self._weighted_geom(a, w)
            if mode == "geom":
                return soft_uw
            if mode == "weighted":
                return soft_w
            if mode == "adaptive_veto":
                return self._adaptive_veto(a, w, soft_uw)
            if mode == "adaptive_veto_trust":
                return self._adaptive_veto_trust(a, soft_uw)
            # adaptive (unweighted soft) / adaptive_weighted (availability-
            # weighted soft): escalate to worst-of when the soft opinion fails.
            soft = soft_uw if mode == "adaptive" else soft_w
            if soft >= self.safety.cfg.adaptive_escalate:
                return soft
            return float(np.min(a))
        return float(np.min(a))      # "min" / default: OR, worst-of

    def _adaptive_veto(self, a: np.ndarray, w: np.ndarray, soft: float) -> float:
        """Asymmetric guardrail: thin data can trigger, but cannot veto.

        Escalation may only be *caused* by a detector that (a) is low and
        (b) had enough data to be credible.  A healthy-but-stale detector (e.g.
        a camera in a feature-poor window retaining its last good score) keeps
        its full voice and can veto escalation.  If no detector is both low and
        credible, we remain at the soft/healthy opinion (do not react on
        unknown).
        """
        escalate = self.safety.cfg.adaptive_escalate
        warn = self.safety.cfg.detector_health_warn
        if soft >= escalate:
            return soft
        credible_low = [a[i] for i in range(len(a))
                        if a[i] < warn and w[i] >= 0.5]
        if not credible_low:
            # Only thin detectors were low -> unknown, not bad.  Keep the best
            # available healthy voice (do not let a thin detector force a hold).
            healthy = [a[i] for i in range(len(a)) if w[i] >= 0.5]
            if healthy:
                return float(max(healthy))
            return float(warn)   # truly fully blind: be neutral/cautious
        return float(min(min(credible_low), soft))

    def _adaptive_veto_trust(self, a: np.ndarray, soft: float) -> float:
        """Like adaptive_veto, but using continuous measured-*trust* instead of
        a binary credibility floor.

        Detectors are ordered (landmark, factorgraph).  A credible low is any
        detector that is (a) below warn and (b) has trust >= 0.45; a low
        opinion from a cluster-of-points camera (many but not spread) is treated
        as thin and cannot trigger.  The healthy veto voice is the highest
        credible detector's score (not a stale thin one).
        """
        escalate = self.safety.cfg.adaptive_escalate
        warn = self.safety.cfg.detector_health_warn
        floor = self.safety.cfg.detector_trust_floor
        if soft >= escalate:
            return soft
        kinds = ("landmark", "factorgraph")
        trusts = np.asarray([self._detector_trust(kinds[i] if i < len(kinds) else "landmark")
                             for i in range(len(a))], dtype=float)
        credible_low = [a[i] for i in range(len(a))
                        if a[i] < warn and trusts[i] >= floor]
        if not credible_low:
            healthy = [a[i] for i in range(len(a)) if trusts[i] >= floor]
            if healthy:
                return float(max(healthy))
            return float(warn)
        return float(min(min(credible_low), soft))

    def _detector_trust(self, kind: str) -> float:
        """A continuous, self-measured *trust* in this frame's geometry.

        Unlike the binary availability weight, this is informative even when a
        detector reports "many" measurements: a camera seeing many landmarks
        that all lie in a tiny angular cluster gives almost no geometric
        leverage, so its opinion on a fault is weak regardless of count.

        landmark  : fractional count  x  angular-diversity (RMS pairwise angle)
        factorgraph: fraction of min factors x convergence (under-determined
                    graph is a weak opinion even if it has many residuals).
        """
        if kind == "landmark":
            dirs = np.asarray(self._landmark_dirs, dtype=float)
            n = self._landmark_obs_count
            if n < 2 or dirs.shape[0] < 2:
                return 0.0
            rms = self._landmark_rms(dirs)
            if self.cfg.trust_learn and self._lm_trust.calibrated:
                tr = self._lm_trust.trust(rms, n)
                if tr is not None:
                    return tr
            # Analytic fallback before/if learning is unavailable: ~69 deg.
            diversity = float(np.clip(rms / 1.2, 0.0, 1.0))
            count_frac = float(np.clip(n / 3.0, 0.0, 1.0))
            return float(count_frac * diversity)
        if kind == "factorgraph":
            if not self.factorgraph_health_trusted:
                return 0.0
            comp = self._fg_flow_components
            base = float(np.clip(comp / max(self.factorgraph.min_keyframes, 1), 0.0, 1.0))
            conv = 1.0 if self._fg_converged else 0.5
            # FG conditioning is already a well-defined structural metric (does
            # the graph have enough factors?), and its reference value is 1.0 by
            # construction, so we keep it analytic rather than learn a baseline
            # from startup samples (which are mostly base=0 while the graph
            # warms up and would weaken the sparse-graph penalty).
            return float(base * conv)
        return 1.0

    @staticmethod
    def _landmark_rms(dirs: np.ndarray) -> float:
        """RMS pairwise angular separation (rad) of the observed body dirs."""
        dirs = np.asarray(dirs, dtype=float)
        n = dirs.shape[0]
        if n < 2:
            return 0.0
        ang = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
                ang.append(float(np.arccos(d)))
        return float(np.sqrt(np.mean(np.square(ang)))) if ang else 0.0

    def _consensus_weights(self, n: int) -> np.ndarray:
        """Per-detector availability weights (landmark, factorgraph)."""
        kinds = ("landmark", "factorgraph")
        w = []
        for i in range(n):
            kind = kinds[i] if i < len(kinds) else "landmark"
            w.append(self._detector_weight(kind))
        w = np.asarray(w, dtype=float)
        # Keep raw availability weights (0..1): _adaptive_veto compares them
        # against a credibility floor.  _weighted_geom normalises internally.
        if w.sum() <= 1e-9:
            return np.ones(n) / n
        return w

    def _weighted_geom(self, a: np.ndarray, w: np.ndarray) -> float:
        eps = 1e-9
        w = np.asarray(w, dtype=float)
        if w.sum() <= 1e-9:
            w = np.ones(len(a)) / len(a)
        else:
            w = w / w.sum()
        raw = np.exp(np.sum(w * np.log(np.clip(a, eps, 1.0))))
        return float(np.clip(raw, 0.0, 1.0))

    def _make_vehicle(self, cfg: SimConfig) -> Quadrotor:
        return Quadrotor(**cfg.vehicle_kwargs)

    def _make_sensors(self, cfg: SimConfig) -> SensorSuite:
        return SensorSuite(**cfg.sensor_kwargs)

    # ---------------- helpers ----------------
    def _sync_truth_estimate(self) -> None:
        """Debug / ablation helper: force the EKF state to ground truth."""
        x = self.ekf.x
        x[S_IDX] = self.vehicle.pos
        x[V_IDX] = self.vehicle.vel
        x[R_IDX] = self.vehicle.rpy
        x[BG_IDX] = 0.0
        x[BA_IDX] = 0.0
        self.ekf.x = _wrap_state(x)

    def _apply_faults(self) -> None:
        if self.cfg.gps_outage is not None:
            start, end = self.cfg.gps_outage
            if start <= self.time < end:
                self.sensors.cfg.gps_dropout = max(self.sensors.cfg.gps_dropout,
                                                   end - self.time)
            else:
                self.sensors.cfg.gps_dropout = 0.0
        self.sensors.cfg.gps_noise_scale = self.cfg.gps_outage_scale

        # velocity-aiding corruption windows / scales
        if self.cfg.flow_outage is not None:
            start, end = self.cfg.flow_outage
            if start <= self.time < end:
                self.sensors.cfg.flow_dropout = max(self.sensors.cfg.flow_dropout,
                                                    end - self.time)
            else:
                self.sensors.cfg.flow_dropout = 0.0
        self.sensors.cfg.flow_scale = self.cfg.flow_scale
        self.sensors.cfg.flow_bias_ramp = self.cfg.flow_bias_ramp

    def _gps_available(self) -> bool:
        return self.sensors.cfg.gps_dropout <= 0.0

    def _base(self) -> np.ndarray:
        return np.asarray(self.cfg.base_pos, dtype=float).reshape(2)

    def _at_base(self) -> bool:
        est = self.ekf.pos
        d = np.linalg.norm(est[:2] - self._base())
        return bool(d <= self.cfg.rtl_radius_m)

    def _can_rtl(self) -> bool:
        """Battery-aware return-to-base feasibility.

        Requires (1) the aircraft to still know *where base is* in world
        coordinates (GNSS up or position uncertainty small), and (2) the
        remaining energy to comfortably cover the cruise back (hover power x
        time) plus the configured reserve.  This keeps the mission-aware
        response from trading a local safe landing for either a mid-return
        power failure or a flight back to an unknown base.
        """
        if not self._gps_available():
            # Without an absolute reference we do not know the base in world
            # frame; RTL would be a guess.  Land where we are.
            return False
        remaining_wh = max(0.0, self.cfg.battery_capacity_wh - self._energy_wh_used)
        dist = float(np.linalg.norm(self.ekf.pos[:2] - self._base()))
        if dist <= self.cfg.rtl_radius_m:
            return True
        speed = max(self.cfg.rtl_speed, 0.1)
        return_time = dist / speed
        return_energy_wh = self.power_model.p_hover * return_time / 3600.0
        reserve_wh = self.cfg.energy_reserve_frac * self.cfg.battery_capacity_wh
        return bool(remaining_wh >= return_energy_wh + reserve_wh)

    def _safety_override(self, pos_ref: np.ndarray, vel_ref: np.ndarray,
                         yaw_ref: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Modify the commanded trajectory based on the safety mode."""
        if mode == CRUISE:
            return pos_ref, vel_ref
        est = self.ekf.pos
        if mode == HOLD:
            return est.copy(), np.zeros(3)
        if mode == RTL:
            # Return to base at current altitude, then the FSM transitions to
            # LAND once the horizontal distance is small enough.
            base = self._base()
            target = np.array([base[0], base[1], est[2]])
            d = np.array([base[0] - est[0], base[1] - est[1]])
            dist = float(np.linalg.norm(d))
            if dist > max(self.cfg.rtl_radius_m, 1e-3):
                v = np.array([d[0] / dist, d[1] / dist, 0.0]) * self.cfg.rtl_speed
            else:
                v = np.zeros(3)
            return target, v
        if mode == LAND:
            # Sit on the ground as the goal, descend at a fixed rate.
            ground = float(self.vehicle.ground)
            target = np.array([est[0], est[1], ground])
            v = np.array([0.0, 0.0, -self.safety.cfg.land_speed])
            return target, v
        if mode == LANDED:
            # Hold on the ground; don't keep commanding descent (which can
            # re-accelerate the vehicle after touchdown).
            ground = float(self.vehicle.ground)
            target = np.array([est[0], est[1], ground])
            return target, np.zeros(3)
        return pos_ref, vel_ref

    def run(self, duration: float | None = None, record: bool = True) -> SimRun:
        duration = duration if duration is not None else self.cfg.duration
        out = SimRun()
        dt = self.cfg.dt
        steps = int(round(duration / dt))

        imu_period = 1.0 / max(self.sensors.cfg.imu_hz, 1.0)
        ahrs_period = 1.0 / max(self.sensors.cfg.ahrs_hz, 1.0)
        baro_period = 1.0 / max(self.sensors.cfg.baro_hz, 1.0)
        gps_period = 1.0 / max(self.sensors.cfg.gps_hz, 1.0)
        mag_period = 1.0 / max(self.sensors.cfg.mag_hz, 1.0)
        flow_period = 1.0 / max(self.sensors.cfg.flow_hz, 1.0)
        landmark_period = 1.0 / max(self.cfg.landmark_hz, 1.0)
        factorgraph_period = 1.0 / max(self.cfg.factorgraph_hz, 1.0)

        t_acc = 0.0
        ah_acc, baro_acc, gps_acc, mag_acc, flow_acc, lm_acc, fg_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(steps):
            self.time = i * dt
            self._apply_faults()
            if self.cfg.truth_estimate:
                self._sync_truth_estimate()

            # ---- uncertainty-aware safety decision (from last EKF state) ----
            gps_ok = self._gps_available()
            # Independent-detector health: combine the *structurally
            # independent* monitors (landmark geometry + factor graph) using the
            # configured consensus policy.  This is the signal that is allowed
            # to force a landing even when GPS is still up, because a corrupt
            # velocity-aiding source drives the controller and can crash the
            # aircraft regardless of an absolute fix.  The factor graph is only
            # trusted once its startup calibration has finished; before that its
            # residual maps to a low, meaningless health and would cause a false
            # hold.
            det_health = None
            det_parts = []
            if self.cfg.landmark_enabled:
                det_parts.append(float(self._landmark_score))
            if self.cfg.factorgraph_enabled:
                fg_health = float(self._factorgraph_health) \
                    if self.factorgraph_health_trusted else 1.0
                det_parts.append(fg_health)
            if det_parts:
                det_health = self._combine_detectors(det_parts)
                # Detector-triggered source rejection: stop trusting the corrupt
                # velocity-aiding source as soon as the independent monitors
                # warn (not only at the LAND threshold).  Rejection must happen
                # *early*: by the time the detector reaches the hard-fail
                # threshold the corrupt velocity has usually already driven the
                # aircraft off course, making any RTL a guess.
                if det_health < self.safety.cfg.detector_health_warn and \
                   not self.flow_rejected:
                    self.flow_rejected = True
                    self._flow_reject_reason = "independent_detector_warn"
                    # If an absolute fix is available, decisively reset the
                    # corrupted position/velocity state so the (trusted) GPS
                    # becomes the source of truth for the subsequent RTL.
                    # Otherwise the contaminated velocity would keep driving
                    # the controller even after the source is rejected.
                    if gps_ok:
                        fix = self.sensors.sample_gps(self.vehicle)
                        if fix is not None:
                            gps_pos, gps_vel = fix
                            gx = self.ekf.x.copy()
                            gx[S_IDX] = np.asarray(gps_pos, dtype=float).reshape(3)
                            gx[V_IDX] = np.asarray(gps_vel, dtype=float).reshape(3)
                            self.ekf.x = _wrap_state(gx)
            dec = self.safety.update(self.ekf, self.sensors,
                                     self.vehicle.altitude, dt, gps_ok,
                                     flow_health=self._effective_flow_health,
                                     detector_health=det_health,
                                     mission_aware=self.cfg.mission_aware,
                                     can_rtl=self._can_rtl(),
                                     at_base=self._at_base())
            self._last_unc_std = dec.uncertainty_std

            # ---- mission desired state using *estimated* position ----
            pos_ref, vel_ref, yaw_ref = self.mission.desired(self.ekf.pos, dt)
            pos_ref, vel_ref = self._safety_override(pos_ref, vel_ref, yaw_ref, dec.mode)

            # ---- controller ----
            control = self.controller.compute(
                self.ekf.pos, self.ekf.vel, self.ekf.rpy,
                pos_ref, vel_ref, yaw_ref,
            )
            # In a true loss-of-trusted-navigation landing, ignore the
            # (unreliable) horizontal position entirely and just level + sink.
            if dec.mode == LAND:
                control = self.controller.landing(
                    self.ekf.rpy, self.ekf.vel,
                    land_speed=self.safety.cfg.land_speed,
                )
            # Once LANDED, ALWAYS use the level-and-hold controller.  It must
            # not fall back to the position cascade above 0.25 m: by that point
            # the velocity estimate is the very thing the detector rejected, so
            # the cascade would drive the aircraft away.  Level + zero descent
            # lets any residual horizontal momentum bleed off on the ground.
            if dec.mode == LANDED:
                control = self.controller.landing(
                    self.ekf.rpy, self.ekf.vel, land_speed=0.0,
                )
            self.last_control = control

            # ---- online energy accounting (drives mission-aware RTL) ----
            if self.cfg.mission_aware:
                specific = float(control[0])
                self._energy_wh_used += self.power_model.power(specific) * dt / 3600.0

            # ---- plant ----
            self.vehicle.step(control, dt, wind_ned=self.cfg.wind_ned)

            # ---- sensor housekeeping + sampling based on schedules ----
            self.sensors.step(dt)
            self.last_imu = self.sensors.sample_imu(self.vehicle, dt)

            t_acc += dt
            ah_acc += dt
            baro_acc += dt
            gps_acc += dt
            mag_acc += dt
            flow_acc += dt
            lm_acc += dt
            fg_acc += dt

            # EKF predict at IMU rate
            self.ekf.predict(self.last_imu.accel, self.last_imu.gyro, dt)

            # Independent IMU dead-reckon for velocity-integrity checking.
            # Remove the *estimated* biases so the drift is tolerable.
            acc_corr = self.last_imu.accel - self.ekf.accel_bias
            gyro_corr = self.last_imu.gyro - self.ekf.gyro_bias
            self.flow_integrity.predict(acc_corr, gyro_corr, dt)
            self._dr_pos += self.flow_integrity.vel * dt
            # Feed the IMU preintegrator from the same corrected measurements.
            R_nb = self.vehicle.R_nb
            acc_ned = R_nb @ acc_corr + np.array([0.0, 0.0, 9.80665])
            self.imu_preint.feed(acc_ned, dt, self.ahrs.rpy)

            if t_acc >= imu_period - 1e-9:
                t_acc = 0.0
            if ah_acc >= ahrs_period - 1e-9:
                self.ahrs.update(self.last_imu.gyro, self.last_imu.accel,
                                 self.sensors.sample_magnetometer(self.vehicle), dt)
                # Treat the AHRS estimate as a low-rate attitude reference.
                self.ekf.update(self.ahrs.rpy, AHRS)
                ah_acc = 0.0
            if baro_acc >= baro_period - 1e-9:
                z = np.array([self.sensors.sample_baro(self.vehicle)])
                self.ekf.update(z, BARO)
                baro_acc = 0.0
            if mag_acc >= mag_period - 1e-9:
                # (magnetometer already used inside AHRS; no separate update)
                mag_acc = 0.0
            if self.cfg.flow_enabled and not self.flow_rejected and flow_acc >= flow_period - 1e-9:
                flow_reading = self.sensors.sample_flow(self.vehicle)
                if flow_reading is not None:
                    self.ekf.update(flow_reading, FLOW)
                    # Health starts from the sensor self-diagnostic (model of
                    # optical-flow confidence / feature quality, noisy).
                    health = float(self.sensors.flow_health)
                    # Independent landmark consistency is the *primary* second
                    # opinion for the velocity measurement itself.
                    # The factor graph is deliberately NOT folded into this
                    # instantaneous flow health: it runs at 2 Hz and its
                    # healthy residual can dip slightly during fast turns.  It
                    # is instead fed to the dedicated independent-detector
                    # safety path, where a wide margin and a longer grace make
                    # it both decisive on real faults and silent on healthy
                    # manouevers.
                    #
                    # Fold the landmark opinion into the velocity-aiding health
                    # ONLY when the camera has enough *geometric leverage* to
                    # be credible.  A degenerate-parallax window (many features
                    # in a tight cone) has low angular-diversity trust and must
                    # not drag down the flow health purely on raw count.
                    if (self.cfg.landmark_enabled and
                            self._detector_trust("landmark") >=
                            self.safety.cfg.detector_trust_floor):
                        health = float(min(health, self._landmark_score))
                    if self.cfg.integrity_monitor_enabled:
                        integrity = self.flow_integrity.evaluate(flow_reading)
                        self._last_flow_mismatch = self.flow_integrity.mismatch
                        health = float(min(health, integrity))
                    self._effective_flow_health = health
                flow_acc = 0.0

            # Independent landmark camera / geometry check.  This runs even
            # when flow is disabled; it only needs the camera + known world.
            if lm_acc >= landmark_period - 1e-9:
                lm_out = self.cfg.landmark_outage
                lm_cluster = self.cfg.landmark_cluster
                if lm_out is not None and lm_out[0] <= self.time < lm_out[1]:
                    # Feature-poor window: no usable landmarks this frame.  The
                    # consistency score holds its last opinion (unknown != bad),
                    # but the observation count drops so the availability weight
                    # gives this detector a weak voice.
                    ids, dirs = np.array([], dtype=int), np.zeros((0, 3))
                else:
                    ids, dirs = self.landmark_field.observe(self.vehicle)
                    # Degenerate-parallax window: keep *many* landmarks but
                    # collapse their bearings into a tight cone.  A camera
                    # looking at a close wall still reports dozens of features,
                    # yet they give almost no geometric leverage.  This is the
                    # fault that count-only trust cannot see.
                    if (lm_cluster is not None and len(ids) >= 2 and
                            lm_cluster[0] <= self.time < lm_cluster[1]):
                        axis = np.mean(np.asarray(dirs, dtype=float), axis=0)
                        norm = float(np.linalg.norm(axis))
                        if norm > 1e-9:
                            axis = axis / norm
                            cone = float(self.cfg.landmark_cluster_cone)
                            # Small deterministic perturbations around the mean
                            # bearing; count stays high, diversity collapses.
                            rng = np.random.default_rng(1234)
                            pert = rng.uniform(-cone, cone, size=(len(ids), 2))
                            dirs = np.array([
                                _cone_bearing(axis, p[0], p[1])
                                for p in pert
                            ], dtype=float)
                self._landmark_obs_count = len(ids)
                self._landmark_dirs = np.asarray(dirs, dtype=float).copy()
                self._landmark_score = self.landmark_consistency.evaluate(
                    ids, dirs, self.ekf.pos, self.ekf.rpy)
                self._landmark_residual = self.landmark_consistency.residual
                # Learn the healthy per-frame distribution before any fault /
                # feature-degeneracy window opens.  Only exclude the frame if we
                # are *inside* an outage/cluster window right now.
                in_out = lm_out is not None and lm_out[0] <= self.time < lm_out[1]
                in_cluster = (lm_cluster is not None and
                              lm_cluster[0] <= self.time < lm_cluster[1])
                if (self.cfg.trust_learn and len(ids) >= 2 and not in_out and
                        not in_cluster and self.time < self.cfg.trust_calibrate_s):
                    self._lm_trust.calibrate(self._landmark_rms(dirs), len(ids))
                lm_acc = 0.0

            # Factor-graph consistency monitor.  Every keyframe we push raw
            # measured data: IMU plus AHRS attitude (for accel->NED), a GNSS
            # fix if available, a flow velocity sample if the sensor is alive,
            # and the camera landmark observation.  The graph then jointly
            # optimises all of these; the post-optimisation flow residual is
            # the independent signal.  If a corrupt flow is consistent with
            # itself but inconsistent with landmarks/IMU/GNSS, that residual
            # will be large and the health will drop.
            if fg_acc >= factorgraph_period - 1e-9:
                if self.cfg.factorgraph_enabled:
                    gps_here = self.sensors.sample_gps(self.vehicle)
                    gps_pos, gps_vel = (gps_here if gps_here is not None
                                        else (None, None))
                    flow_here = None
                    if self.cfg.flow_enabled:
                        fg_flow_out = self.cfg.factorgraph_flow_outage
                        in_fg_flow_out = (fg_flow_out is not None and
                                          fg_flow_out[0] <= self.time < fg_flow_out[1])
                        if not in_fg_flow_out:
                            flow_here = self.sensors.sample_flow(self.vehicle)
                        # else: sparse graph — no flow factor this frame.
                    ids_here, dirs_here = self.landmark_field.observe(self.vehicle)
                    # Seed the graph from the IMU preintegrator (clean relative
                    # pose since the last trusted anchor), NOT the flow-fed EKF.
                    # Pass RAW accelerometer (no EKF bias subtraction): the
                    # graph estimates its own bias from the independent
                    # landmark/GPS geometry, so a corrupt flow cannot
                    # contaminate the IMU model.
                    kf = build_keyframe(
                        self.imu_preint.predict_pos(),
                        self.imu_preint.predict_vel(),
                        self.last_imu.accel,
                        self.ahrs.rpy,
                        factorgraph_period,
                        flow_here, gps_pos, gps_vel,
                        ids_here, dirs_here,
                    )
                    self.factorgraph.push(kf)
                    fg_info = self.factorgraph.optimize()
                    self._factorgraph_health = fg_info["health"]
                    self._factorgraph_residual = fg_info["flow_residual"]
                    self._fg_flow_components = int(fg_info.get("flow_components", 0))
                    self._fg_converged = bool(fg_info.get("converged", False))
                    # Anchor the preintegrator to the graph's optimized latest
                    # pose so the next relative prediction is correct.
                    if fg_info.get("converged") and len(self.factorgraph.keyframes):
                        last = self.factorgraph.keyframes[-1]
                        self.imu_preint.reset(last.p0, last.v0)

                    # Startup calibration: while GNSS is available and we have
                    # a trusted absolute reference, collect the residual of a
                    # *healthy* mission and use a robust high percentile as the
                    # baseline.  Using the mean underestimates the healthy
                    # motion floor and causes false HOLD during turns; using a
                    # robust high percentile makes a healthy flight map to
                    # health~1.0 while a real fault (residual >> floor) is
                    # still clearly detected.
                    # FG conditioning is kept analytic (see _detector_trust);
                    # no per-frame learning here.

                    if not self._fg_calibrated and gps_ok:
                        self._fg_calibration_samples.append(fg_info["flow_residual"])
                        if len(self._fg_calibration_samples) >= 6:
                            samples = np.sort(np.asarray(self._fg_calibration_samples))
                            idx = int(0.9 * (len(samples) - 1))
                            self._fg_baseline = float(samples[idx])
                            self.factorgraph.baseline_residual = self._fg_baseline
                            self._fg_calibrated = True
                            self.factorgraph_health_trusted = True
                fg_acc = 0.0

            if gps_acc >= gps_period - 1e-9:
                gps_reading = self.sensors.sample_gps(self.vehicle)
                if gps_reading is not None:
                    pos_g, vel_g = gps_reading
                    self.ekf.update(np.concatenate([pos_g, vel_g]), GPS)
                    # Re-anchor the independent dead-reckon to the corrected
                    # state whenever a trusted absolute fix is available.
                    self.flow_integrity.reset(self.ekf.vel, self.ekf.rpy)
                    self._dr_pos = self.ekf.pos.copy()
                    self._dr_vel = self.ekf.vel.copy()
                    # Re-anchor the preintegrator so its relative prediction
                    # does not carry an old start-pose drift forward.
                    self.imu_preint.reset(self.ekf.pos, self.ekf.vel)
                gps_acc = 0.0

            if record:
                out.record(self, pos_ref, vel_ref, yaw_ref)

        return out
