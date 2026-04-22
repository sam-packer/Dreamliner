"""Helpers for instantiating JSBSim and applying stall-recovery initial conditions."""

from __future__ import annotations

import atexit
import re
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Mapping, Sequence

import jsbsim
import numpy as np


def jsbsim_root() -> str:
    """Return the path to the bundled jsbsim data directory (aircraft, engines, ...)."""
    with resources.as_file(resources.files("jsbsim")) as p:
        return str(p)


# --- Aircraft XML patching ---------------------------------------------------
# JSBSim's bundled 737.xml declares <input port="..."> sockets (telnet 5137,
# QTJSBSIM 5139). When >1 FDM exists in the same process - or in any process
# on the host - the second binder onward prints "Could not bind to TCP/UDP
# input socket". We don't use the JSBSim control sockets, so we ship a
# patched copy of the aircraft dir with those declarations removed.

_PATCHED_AIRCRAFT_ROOT: Path | None = None

# Match <input ... port="..."> declarations only. Plain <input>foo</input>
# elements inside FCS components have no port= attribute and stay intact.
_INPUT_SOCKET_SELF_CLOSING = re.compile(r'<input\s+[^>]*\bport\s*=\s*"[^"]*"[^>]*?/>', re.DOTALL)
_INPUT_SOCKET_PAIRED = re.compile(
    r'<input\s+[^>]*\bport\s*=\s*"[^"]*"[^>]*?>.*?</input>', re.DOTALL,
)


def _strip_input_sockets(xml: str) -> str:
    xml = _INPUT_SOCKET_PAIRED.sub("", xml)
    xml = _INPUT_SOCKET_SELF_CLOSING.sub("", xml)
    return xml


def _patched_aircraft_path(aircraft: str) -> Path:
    """Return a search path containing a patched copy of `aircraft/`."""
    global _PATCHED_AIRCRAFT_ROOT
    if _PATCHED_AIRCRAFT_ROOT is None:
        _PATCHED_AIRCRAFT_ROOT = Path(tempfile.mkdtemp(prefix="dreamliner_aircraft_"))
        atexit.register(shutil.rmtree, _PATCHED_AIRCRAFT_ROOT, ignore_errors=True)

    target_dir = _PATCHED_AIRCRAFT_ROOT / aircraft
    if not target_dir.exists():
        src_dir = Path(jsbsim_root()) / "aircraft" / aircraft
        shutil.copytree(src_dir, target_dir)
        main_xml = target_dir / f"{aircraft}.xml"
        main_xml.write_text(_strip_input_sockets(main_xml.read_text(encoding="utf-8")), encoding="utf-8")
    return _PATCHED_AIRCRAFT_ROOT


# --- Property name constants -------------------------------------------------
# The 737 model lacks `fcs/aileron-pos-norm`; we use left/right and average.

class P:
    # State
    alpha_deg          = "aero/alpha-deg"
    beta_deg           = "aero/beta-deg"
    stall_hyst_norm    = "aero/stall-hyst-norm"
    pitch_rad          = "attitude/pitch-rad"
    roll_rad           = "attitude/roll-rad"
    heading_rad        = "attitude/heading-true-rad"
    vc_kts             = "velocities/vc-kts"
    v_down_fps         = "velocities/v-down-fps"
    p_rad_sec          = "velocities/p-rad_sec"
    q_rad_sec          = "velocities/q-rad_sec"
    r_rad_sec          = "velocities/r-rad_sec"
    altitude_ft        = "position/h-sl-ft"
    load_factor_g      = "accelerations/n-pilot-z-norm"
    thrust_lbs_eng0    = "propulsion/engine[0]/thrust-lbs"

    # Surface positions (the 737 model only exposes per-side aileron pos)
    elevator_pos_norm  = "fcs/elevator-pos-norm"
    rudder_pos_norm    = "fcs/rudder-pos-norm"
    left_aileron_norm  = "fcs/left-aileron-pos-norm"
    right_aileron_norm = "fcs/right-aileron-pos-norm"

    # Control commands
    elevator_cmd       = "fcs/elevator-cmd-norm"
    aileron_cmd        = "fcs/aileron-cmd-norm"
    rudder_cmd         = "fcs/rudder-cmd-norm"
    throttle_cmd       = "fcs/throttle-cmd-norm"
    gear_cmd_norm      = "gear/gear-cmd-norm"
    gear_pos_norm      = "gear/gear-pos-norm"

    # Initial conditions
    ic_h_sl_ft         = "ic/h-sl-ft"
    ic_vc_kts          = "ic/vc-kts"
    ic_alpha_deg       = "ic/alpha-deg"
    ic_beta_deg        = "ic/beta-deg"
    ic_lat_geod_deg    = "ic/lat-geod-deg"
    ic_long_gc_deg     = "ic/long-gc-deg"
    ic_phi_deg         = "ic/phi-deg"
    ic_theta_deg       = "ic/theta-deg"
    ic_psi_true_deg    = "ic/psi-true-deg"
    ic_p_rad_sec       = "ic/p-rad_sec"
    ic_q_rad_sec       = "ic/q-rad_sec"
    ic_r_rad_sec       = "ic/r-rad_sec"


# --- FDM construction --------------------------------------------------------

def make_fdm(
    aircraft: str,
    sim_dt_hz: int,
    output_directive: str | Path | None = None,
) -> jsbsim.FGFDMExec:
    """Construct an FGFDMExec with the given aircraft loaded and dt configured.

    If ``output_directive`` is set, point JSBSim at that XML file (e.g. our
    ``flightgear.xml``) and turn the output channel on.
    """
    try:
        aircraft_path = _patched_aircraft_path(aircraft)
        fdm = jsbsim.FGFDMExec(root_dir=jsbsim_root())
        fdm.set_debug_level(0)
        fdm.set_aircraft_path(str(aircraft_path))
        fdm.set_dt(1.0 / sim_dt_hz)
        if not fdm.load_model(aircraft):
            raise RuntimeError(
                f"FGFDMExec.load_model({aircraft!r}) returned False. "
                f"Aircraft search path: {aircraft_path}. "
                f"Verify {aircraft_path / aircraft / f'{aircraft}.xml'} exists."
            )
        if output_directive is not None:
            fdm.set_output_directive(str(output_directive))
            fdm.enable_output()
        return fdm
    except Exception as e:
        # Worker-subprocess errors otherwise surface as opaque BrokenPipe in the
        # parent. Re-raise with the JSBSim context attached.
        raise RuntimeError(
            f"make_fdm failed (aircraft={aircraft!r}, sim_dt_hz={sim_dt_hz}, "
            f"jsbsim_root={jsbsim_root()}): {type(e).__name__}: {e}"
        ) from e


def flightgear_directive_path() -> Path:
    """Path to the bundled FlightGear UDP output directive (port 5550, 60 Hz)."""
    return Path(__file__).resolve().parent.parent / "data" / "flightgear.xml"


# --- Stall scenarios ---------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    weight: float
    altitude_ft: tuple[float, float]
    airspeed_kcas: tuple[float, float]
    alpha_deg: tuple[float, float]
    pitch_deg: tuple[float, float]
    roll_deg: tuple[float, float]      # absolute range; sign randomized for turning/spin
    beta_deg: tuple[float, float]
    yaw_rate_dps: tuple[float, float]  # signed for spin
    throttle: tuple[float, float]


def parse_scenarios(raw: Mapping[str, Mapping]) -> list[ScenarioSpec]:
    out: list[ScenarioSpec] = []
    for name, cfg in raw.items():
        out.append(ScenarioSpec(
            name=name,
            weight=float(cfg.get("weight", 1.0)),
            altitude_ft=tuple(cfg["altitude_ft"]),
            airspeed_kcas=tuple(cfg["airspeed_kcas"]),
            alpha_deg=tuple(cfg["alpha_deg"]),
            pitch_deg=tuple(cfg["pitch_deg"]),
            roll_deg=tuple(cfg["roll_deg"]),
            beta_deg=tuple(cfg["beta_deg"]),
            yaw_rate_dps=tuple(cfg["yaw_rate_dps"]),
            throttle=tuple(cfg["throttle"]),
        ))
    return out


def sample_scenario(rng: np.random.Generator, scenarios: Sequence[ScenarioSpec]) -> ScenarioSpec:
    weights = np.array([s.weight for s in scenarios], dtype=np.float64)
    weights = weights / weights.sum()
    idx = int(rng.choice(len(scenarios), p=weights))
    return scenarios[idx]


# --- Curriculum scheduling ---------------------------------------------------

@dataclass(frozen=True)
class CurriculumPhase:
    start_step: int
    weights: Mapping[str, float]


class CurriculumSchedule:
    """Step-based scenario weight schedule.

    Reads the current global step from a file written by the trainer, then
    picks the latest phase whose ``start_step <= step`` and samples a scenario
    using that phase's weights. Missing / unreadable step file -> phase 0.
    """

    def __init__(self, phases: Sequence[CurriculumPhase], step_file: Path | None):
        self._phases: list[CurriculumPhase] = sorted(phases, key=lambda p: p.start_step)
        if not self._phases or self._phases[0].start_step != 0:
            raise ValueError("Curriculum must define a phase at start_step=0")
        self._step_file = step_file

    def current_step(self) -> int:
        if self._step_file is None:
            return 0
        try:
            text = self._step_file.read_text(encoding="utf-8").strip()
            return int(text) if text else 0
        except (OSError, ValueError):
            return 0

    def active_phase(self, step: int) -> CurriculumPhase:
        active = self._phases[0]
        for phase in self._phases:
            if step >= phase.start_step:
                active = phase
            else:
                break
        return active

    def sample(
        self,
        rng: np.random.Generator,
        scenarios: Sequence[ScenarioSpec],
    ) -> tuple[ScenarioSpec, int]:
        """Sample a scenario under the currently-active phase. Returns (scenario, current_step)."""
        step = self.current_step()
        phase = self.active_phase(step)
        weights = np.array(
            [float(phase.weights.get(s.name, s.weight)) for s in scenarios],
            dtype=np.float64,
        )
        total = weights.sum()
        if total <= 0.0:
            raise ValueError(
                f"Curriculum phase at start_step={phase.start_step} has non-positive total weight"
            )
        weights /= total
        idx = int(rng.choice(len(scenarios), p=weights))
        return scenarios[idx], step


def parse_curriculum(raw: Mapping | None) -> tuple[bool, list[CurriculumPhase]]:
    """Parse a ``curriculum`` config block. Returns ``(enabled, phases)``.

    Disabled / missing -> ``(False, [])``. When enabled, the first phase must
    be at step 0 (enforced by CurriculumSchedule.__init__).
    """
    if not raw or not raw.get("enabled", False):
        return False, []
    phases: list[CurriculumPhase] = []
    for p in raw.get("phases", []):
        phases.append(CurriculumPhase(
            start_step=int(p["start_step"]),
            weights=dict(p["weights"]),
        ))
    return True, phases


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def apply_initial_conditions(
    fdm: jsbsim.FGFDMExec,
    ics: Mapping[str, float],
    *,
    location: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Write exact initial conditions to ``fdm`` and call ``run_ic()``."""
    altitude = float(ics["altitude_ft"])
    airspeed = float(ics["airspeed_kcas"])
    alpha = float(ics["alpha_deg"])
    pitch = float(ics["pitch_deg"])
    roll = float(ics["roll_deg"])
    beta = float(ics["beta_deg"])
    yaw_rate = float(ics["yaw_rate_dps"])
    throttle = float(ics["throttle"])

    fdm[P.ic_h_sl_ft] = altitude
    fdm[P.ic_vc_kts] = airspeed
    fdm[P.ic_alpha_deg] = alpha
    fdm[P.ic_theta_deg] = pitch
    fdm[P.ic_phi_deg] = roll
    fdm[P.ic_beta_deg] = beta
    if location is not None:
        latitude_deg, longitude_deg = location
        fdm[P.ic_lat_geod_deg] = latitude_deg
        fdm[P.ic_long_gc_deg] = longitude_deg
    fdm[P.ic_psi_true_deg] = 0.0
    fdm[P.ic_p_rad_sec] = 0.0
    fdm[P.ic_q_rad_sec] = 0.0
    fdm[P.ic_r_rad_sec] = np.deg2rad(yaw_rate)

    if not fdm.run_ic():
        raise RuntimeError("fdm.run_ic() returned False; ICs may be inconsistent.")

    # Set throttle and start the engines so we have thrust available immediately.
    fdm[P.throttle_cmd] = throttle
    fdm.get_propulsion().init_running(-1)  # -1 = all engines

    return {
        "altitude_ft": altitude,
        "airspeed_kcas": airspeed,
        "alpha_deg": alpha,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "beta_deg": beta,
        "yaw_rate_dps": yaw_rate,
        "throttle": throttle,
    }


def apply_scenario(
    fdm: jsbsim.FGFDMExec,
    scenario: ScenarioSpec,
    rng: np.random.Generator,
    *,
    location: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Sample ICs from `scenario`, write them to `fdm`, and call run_ic().

    Returns the sampled ICs (for logging / observation start state).
    """
    altitude = _u(rng, *scenario.altitude_ft)
    airspeed = _u(rng, *scenario.airspeed_kcas)
    alpha    = _u(rng, *scenario.alpha_deg)
    pitch    = _u(rng, *scenario.pitch_deg)
    beta     = _u(rng, *scenario.beta_deg)
    throttle = _u(rng, *scenario.throttle)

    # Roll: sign randomized for scenarios that allow either direction.
    roll_mag = _u(rng, *scenario.roll_deg)
    if scenario.name in ("turning_stall", "incipient_spin"):
        roll_mag *= rng.choice([-1.0, 1.0])

    # Yaw rate: incipient spin should commit to a sign; others are signed in YAML range.
    if scenario.name == "incipient_spin":
        yaw_rate = _u(rng, *scenario.yaw_rate_dps) * rng.choice([-1.0, 1.0])
    else:
        yaw_rate = _u(rng, *scenario.yaw_rate_dps)

    return apply_initial_conditions(
        fdm,
        {
            "altitude_ft": altitude,
            "airspeed_kcas": airspeed,
            "alpha_deg": alpha,
            "pitch_deg": pitch,
            "roll_deg": roll_mag,
            "beta_deg": beta,
            "yaw_rate_dps": yaw_rate,
            "throttle": throttle,
        },
        location=location,
    )


def aileron_pos_norm(fdm: jsbsim.FGFDMExec) -> float:
    """The 737 model lacks `fcs/aileron-pos-norm`; average left/right surfaces."""
    return 0.5 * (fdm[P.left_aileron_norm] - fdm[P.right_aileron_norm])


# --- Config loading ----------------------------------------------------------

def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
