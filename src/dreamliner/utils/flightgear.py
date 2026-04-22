"""FlightGear discovery + launching.

The Windows installer doesn't add ``fgfs.exe`` to ``PATH``, so we look in the
standard install locations. ``launch_flightgear()`` spawns it detached so the
cockpit window survives even after ``play`` exits.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jsbsim

# Default visual: Boeing 737-300. Download it once via the FlightGear launcher
# (Aircraft tab -> search "737-300" -> Install) before running. JSBSim drives
# the actual flight dynamics through the native-FDM socket; the FG aircraft is
# the visual / cockpit only.
_DEFAULT_AIRCRAFT = "737-300"
_FG_FDM_HOST = "127.0.0.1"
_FG_FDM_PORT = 5550
_FG_NET_FDM_VERSION = 24
_FG_MAX_ENGINES = 4
_FG_MAX_TANKS = 4
_FG_MAX_WHEELS = 3
_FG_NATIVE_FDM_RATE_HZ = 120
_FG_PROPERTY_SYNC_RATE_HZ = 60
_FG_MAX_FPS = 144
_FG_TEXTURE_FILTERING = 16
# Keep visuals on a stable UTC location so a fixed unix timestamp reliably means afternoon.
_FIXED_VISUAL_LATITUDE_DEG = 5.605189
_FIXED_VISUAL_LONGITUDE_DEG = -0.166786
_FIXED_AFTERNOON_UNIX_TIME = int(datetime(2026, 6, 21, 15, 20, 0, tzinfo=timezone.utc).timestamp())
_FG_NET_FDM_FORMAT = (
    "!"
    "II"
    "ddd"
    + "f" * 22
    + "I"
    + "I" * _FG_MAX_ENGINES
    + "f" * (_FG_MAX_ENGINES * 9)
    + "I"
    + "f" * _FG_MAX_TANKS
    + "I"
    + "I" * _FG_MAX_WHEELS
    + "f" * (_FG_MAX_WHEELS * 3)
    + "Ii"
    + "f" * 11
)

# FG's telnet-props interface. We query /sim/sceneryloaded to know when the
# cockpit is actually ready to receive FDM packets (scenery load is the slow
# step — often 30-60s on first launch).
_TELNET_PORT = 5501
_TELNET_TIMEOUT_SECS = 2.0
_VIEW_ALIAS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cockpit": ("Pilot View", "Captain View", "Captain", "Cockpit View", "Pilot"),
    "chase": ("Chase View", "Helicopter View", "Chase"),
    "tower": ("Tower View", "Tower"),
    "fly-by": ("Fly-By View", "Fly By View", "Fly-by"),
}

_ALPHA_RAD = "aero/alpha-rad"
_BETA_RAD = "aero/beta-rad"
_BETA_DEG = "aero/beta-deg"
_ALTITUDE_AGL_FT = "position/h-agl-ft"
_LATITUDE_RAD = "position/lat-geod-rad"
_LONGITUDE_RAD = "position/long-gc-rad"
_PILOT_ACCEL_X = "accelerations/a-pilot-x-ft_sec2"
_PILOT_ACCEL_Y = "accelerations/a-pilot-y-ft_sec2"
_PILOT_ACCEL_Z = "accelerations/a-pilot-z-ft_sec2"
_BODY_U_FPS = "velocities/u-fps"
_BODY_V_FPS = "velocities/v-fps"
_BODY_W_FPS = "velocities/w-fps"
_VEL_NORTH_FPS = "velocities/v-north-fps"
_VEL_EAST_FPS = "velocities/v-east-fps"
_ELEVATOR_CMD_NORM = "fcs/elevator-cmd-norm"
_AILERON_CMD_NORM = "fcs/aileron-cmd-norm"
_RUDDER_CMD_NORM = "fcs/rudder-cmd-norm"
_ELEVATOR_POS_NORM = "fcs/elevator-pos-norm"
_LEFT_AILERON_POS_NORM = "fcs/left-aileron-pos-norm"
_RIGHT_AILERON_POS_NORM = "fcs/right-aileron-pos-norm"
_RUDDER_POS_NORM = "fcs/rudder-pos-norm"
_FLAP_POS_NORM = "fcs/flap-pos-norm"
_SPEEDBRAKE_POS_NORM = "fcs/speedbrake-pos-norm"
_SPOILER_POS_NORM = "fcs/spoiler-pos-norm"
_PITCH_TRIM_CMD_NORM = "fcs/pitch-trim-cmd-norm"
_GEAR_WOW = "gear/unit[{idx}]/WOW"
_GEAR_POS_NORM = "gear/gear-pos-norm"
_GEAR_COMPRESSION_FT = "gear/unit[{idx}]/compression-ft"
_ENGINE_N1 = "propulsion/engine/n1"
_ENGINE_N2 = "propulsion/engine/n2"
_ENGINE_THRUST_LBS = "propulsion/engine/thrust-lbs"
_ENGINE_FUEL_FLOW_GPH = "propulsion/engine/fuel-flow-rate-gph"
_TANK_CONTENTS_LBS = "propulsion/tank/contents-lbs"


def _engine_property(base: str, idx: int) -> str:
    return base if idx == 0 else f"{base}[{idx}]"


def _tank_property(base: str, idx: int) -> str:
    return base if idx == 0 else f"{base}[{idx}]"


def _get_float_property(fdm: jsbsim.FGFDMExec, path: str, default: float = 0.0) -> float:
    try:
        return float(fdm[path])
    except KeyError:
        return default


def _count_present_properties(
    fdm: jsbsim.FGFDMExec,
    property_factory,
    limit: int,
) -> int:
    count = 0
    for idx in range(limit):
        try:
            fdm[property_factory(idx)]
        except KeyError:
            break
        count += 1
    return count


class FlightGearPropertySyncClient:
    """Best-effort sync for FG properties that addon aircraft animate directly.

    Native-FDM covers pose and control-surface deflections, but complex addon
    aircraft also animate cockpit controls and some systems from property-tree
    nodes outside the net_fdm packet. We mirror the few nodes the 737-300 uses
    so those visuals stop fighting FlightGear's local defaults.
    """

    def __init__(
        self,
        *,
        host: str = _FG_FDM_HOST,
        port: int = _TELNET_PORT,
        timeout: float = 0.15,
        reconnect_delay: float = 1.0,
    ) -> None:
        self._endpoint = (host, port)
        self._timeout = timeout
        self._reconnect_delay = reconnect_delay
        self._sock: socket.socket | None = None
        self._next_connect_time = 0.0

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def send(self, fdm: jsbsim.FGFDMExec) -> None:
        sock = self._ensure_connected()
        if sock is None:
            return
        try:
            sock.sendall(self._build_payload(fdm))
            self._drain_input(sock)
        except OSError:
            self.close()
            self._next_connect_time = time.monotonic() + self._reconnect_delay

    def _ensure_connected(self) -> socket.socket | None:
        if self._sock is not None:
            return self._sock
        if time.monotonic() < self._next_connect_time:
            return None
        try:
            sock = socket.create_connection(self._endpoint, timeout=self._timeout)
            sock.settimeout(self._timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
        except OSError:
            self._next_connect_time = time.monotonic() + self._reconnect_delay
            return None
        return self._sock

    def _drain_input(self, sock: socket.socket) -> None:
        sock.setblocking(False)
        try:
            while True:
                try:
                    chunk = sock.recv(4096)
                except BlockingIOError:
                    break
                if not chunk:
                    self.close()
                    break
        finally:
            if self._sock is not None:
                sock.setblocking(True)
                sock.settimeout(self._timeout)

    def _build_payload(self, fdm: jsbsim.FGFDMExec) -> bytes:
        flap_pos = _get_float_property(fdm, _FLAP_POS_NORM)
        speedbrake_pos = _get_float_property(fdm, _SPEEDBRAKE_POS_NORM)
        spoiler_pos = _get_float_property(fdm, _SPOILER_POS_NORM)
        spoiler_lever = int(round(max(speedbrake_pos, spoiler_pos) * 6.0))
        properties = (
            ("/controls/flight/elevator", _get_float_property(fdm, _ELEVATOR_CMD_NORM)),
            ("/controls/flight/aileron", _get_float_property(fdm, _AILERON_CMD_NORM)),
            ("/controls/flight/rudder", _get_float_property(fdm, _RUDDER_CMD_NORM)),
            ("/controls/flight/elevator-trim", _get_float_property(fdm, _PITCH_TRIM_CMD_NORM)),
            ("/controls/flight/flaps", flap_pos),
            ("/controls/flight/speedbrake", speedbrake_pos),
            ("/controls/flight/spoilers", spoiler_pos),
            ("/b733/controls/flight/spoilers-lever-pos", spoiler_lever),
            ("/surface-positions/elevator-pos-norm", _get_float_property(fdm, _ELEVATOR_POS_NORM)),
            ("/surface-positions/left-aileron-pos-norm", _get_float_property(fdm, _LEFT_AILERON_POS_NORM)),
            ("/surface-positions/right-aileron-pos-norm", _get_float_property(fdm, _RIGHT_AILERON_POS_NORM)),
            ("/surface-positions/rudder-pos-norm", _get_float_property(fdm, _RUDDER_POS_NORM)),
            ("/surface-positions/flap-pos-norm", flap_pos),
            ("/surface-positions/flap-pos-norm[0]", flap_pos),
            ("/surface-positions/flap-pos-norm[1]", flap_pos),
            ("/surface-positions/speedbrake-pos-norm", speedbrake_pos),
            ("/surface-positions/spoilers-pos-norm", spoiler_pos),
        )
        lines: list[str] = []
        for path, value in properties:
            if isinstance(value, int):
                lines.append(f"set {path} {value}\r\n")
            else:
                lines.append(f"set {path} {value:.6f}\r\n")
        return "".join(lines).encode("utf-8")


class FlightGearNativeFDMClient:
    """Direct native-FDM sender used instead of JSBSim's built-in FG output."""

    def __init__(
        self,
        *,
        sim_dt_hz: int,
        host: str = _FG_FDM_HOST,
        port: int = _FG_FDM_PORT,
        rate_hz: int = _FG_NATIVE_FDM_RATE_HZ,
        property_sync_rate_hz: int = _FG_PROPERTY_SYNC_RATE_HZ,
    ) -> None:
        if sim_dt_hz % rate_hz != 0:
            raise ValueError("FlightGear send rate must divide sim_dt_hz exactly")
        if rate_hz % property_sync_rate_hz != 0:
            raise ValueError("FlightGear property sync rate must divide send rate exactly")
        self._send_every_substeps = sim_dt_hz // rate_hz
        self._substep_counter = 0
        self._sent_packets = 0
        self._property_sync_every_packets = rate_hz // property_sync_rate_hz
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._target = (host, port)
        self._property_sync = FlightGearPropertySyncClient()

    @property
    def latitude_deg(self) -> float:
        return _FIXED_VISUAL_LATITUDE_DEG

    @property
    def longitude_deg(self) -> float:
        return _FIXED_VISUAL_LONGITUDE_DEG

    def reset(self) -> None:
        self._substep_counter = 0
        self._sent_packets = 0

    def send_initial(self, fdm: jsbsim.FGFDMExec) -> None:
        self.reset()
        self._send(fdm)

    def maybe_send(self, fdm: jsbsim.FGFDMExec) -> None:
        self._substep_counter += 1
        if self._substep_counter % self._send_every_substeps == 0:
            self._send(fdm)

    def close(self) -> None:
        self._property_sync.close()
        self._sock.close()

    def _send(self, fdm: jsbsim.FGFDMExec) -> None:
        self._sock.sendto(self._build_packet(fdm), self._target)
        if self._sent_packets % self._property_sync_every_packets == 0:
            self._property_sync.send(fdm)
        self._sent_packets += 1

    def _build_packet(self, fdm: jsbsim.FGFDMExec) -> bytes:
        phi = _get_float_property(fdm, "attitude/roll-rad")
        theta = _get_float_property(fdm, "attitude/pitch-rad")
        psi = _get_float_property(fdm, "attitude/heading-true-rad")
        p = _get_float_property(fdm, "velocities/p-rad_sec")
        q = _get_float_property(fdm, "velocities/q-rad_sec")
        r = _get_float_property(fdm, "velocities/r-rad_sec")

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        cos_theta = math.cos(theta)
        mix = q * sin_phi + r * cos_phi
        phi_dot = p + math.tan(theta) * mix
        theta_dot = q * cos_phi - r * sin_phi
        psi_dot = mix / cos_theta if abs(cos_theta) > 1e-6 else 0.0

        propulsion = fdm.get_propulsion()
        num_engines = min(_FG_MAX_ENGINES, int(propulsion.get_num_engines()))
        eng_state = [0] * _FG_MAX_ENGINES
        rpm = [0.0] * _FG_MAX_ENGINES
        fuel_flow = [0.0] * _FG_MAX_ENGINES
        fuel_px = [0.0] * _FG_MAX_ENGINES
        egt = [0.0] * _FG_MAX_ENGINES
        cht = [0.0] * _FG_MAX_ENGINES
        mp_osi = [0.0] * _FG_MAX_ENGINES
        tit = [0.0] * _FG_MAX_ENGINES
        oil_temp = [0.0] * _FG_MAX_ENGINES
        oil_px = [0.0] * _FG_MAX_ENGINES
        for idx in range(num_engines):
            n1 = _get_float_property(fdm, _engine_property(_ENGINE_N1, idx))
            n2 = _get_float_property(fdm, _engine_property(_ENGINE_N2, idx))
            thrust = _get_float_property(fdm, _engine_property(_ENGINE_THRUST_LBS, idx))
            fuel_flow[idx] = _get_float_property(fdm, _engine_property(_ENGINE_FUEL_FLOW_GPH, idx))
            if n1 > 1.0 or n2 > 1.0 or thrust > 1.0:
                eng_state[idx] = 2

        num_tanks = min(
            _FG_MAX_TANKS,
            _count_present_properties(fdm, lambda idx: _tank_property(_TANK_CONTENTS_LBS, idx), _FG_MAX_TANKS),
        )
        fuel_quantity = [0.0] * _FG_MAX_TANKS
        for idx in range(num_tanks):
            fuel_quantity[idx] = _get_float_property(fdm, _tank_property(_TANK_CONTENTS_LBS, idx))

        ground_reactions = fdm.get_ground_reactions()
        num_wheels = min(_FG_MAX_WHEELS, int(ground_reactions.get_num_gear_units()))
        wow = [0] * _FG_MAX_WHEELS
        gear_pos = [0.0] * _FG_MAX_WHEELS
        gear_steer = [0.0] * _FG_MAX_WHEELS
        gear_compression = [0.0] * _FG_MAX_WHEELS
        gear_pos_norm = _get_float_property(fdm, _GEAR_POS_NORM)
        for idx in range(num_wheels):
            wow[idx] = int(_get_float_property(fdm, _GEAR_WOW.format(idx=idx)))
            gear_pos[idx] = gear_pos_norm
            gear_steer[idx] = float(ground_reactions.get_gear_unit(idx).get_steer_norm())
            gear_compression[idx] = _get_float_property(fdm, _GEAR_COMPRESSION_FT.format(idx=idx))

        fields: list[float | int] = [
            _FG_NET_FDM_VERSION,
            0,
            _get_float_property(fdm, _LONGITUDE_RAD),
            _get_float_property(fdm, _LATITUDE_RAD),
            _get_float_property(fdm, "position/h-sl-ft") * 0.3048,
            _get_float_property(fdm, _ALTITUDE_AGL_FT) * 0.3048,
            phi,
            theta,
            psi,
            _get_float_property(fdm, _ALPHA_RAD),
            _get_float_property(fdm, _BETA_RAD),
            phi_dot,
            theta_dot,
            psi_dot,
            _get_float_property(fdm, "velocities/vc-kts"),
            -_get_float_property(fdm, "velocities/v-down-fps"),
            _get_float_property(fdm, _VEL_NORTH_FPS),
            _get_float_property(fdm, _VEL_EAST_FPS),
            _get_float_property(fdm, "velocities/v-down-fps"),
            _get_float_property(fdm, _BODY_U_FPS),
            _get_float_property(fdm, _BODY_V_FPS),
            _get_float_property(fdm, _BODY_W_FPS),
            _get_float_property(fdm, _PILOT_ACCEL_X),
            _get_float_property(fdm, _PILOT_ACCEL_Y),
            _get_float_property(fdm, _PILOT_ACCEL_Z),
            0.0,
            _get_float_property(fdm, _BETA_DEG),
            num_engines,
            *eng_state,
            *rpm,
            *fuel_flow,
            *fuel_px,
            *egt,
            *cht,
            *mp_osi,
            *tit,
            *oil_temp,
            *oil_px,
            num_tanks,
            *fuel_quantity,
            num_wheels,
            *wow,
            *gear_pos,
            *gear_steer,
            *gear_compression,
            _FIXED_AFTERNOON_UNIX_TIME,
            0,
            25000.0,
            _get_float_property(fdm, _ELEVATOR_POS_NORM),
            _get_float_property(fdm, _PITCH_TRIM_CMD_NORM),
            _get_float_property(fdm, _FLAP_POS_NORM),
            _get_float_property(fdm, _FLAP_POS_NORM),
            _get_float_property(fdm, _LEFT_AILERON_POS_NORM),
            _get_float_property(fdm, _RIGHT_AILERON_POS_NORM),
            _get_float_property(fdm, _RUDDER_POS_NORM),
            _get_float_property(fdm, _RUDDER_POS_NORM),
            _get_float_property(fdm, _SPEEDBRAKE_POS_NORM),
            _get_float_property(fdm, _SPOILER_POS_NORM),
        ]
        return struct.pack(_FG_NET_FDM_FORMAT, *fields)


def _normalize_view_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("-", " ").split())


def _telnet_command(
    command: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
    expect_response: bool = True,
) -> str:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(command.encode("utf-8") + b"\r\n")
        if not expect_response:
            return ""
        buf = b""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = s.recv(4096)
            except (TimeoutError, socket.timeout):
                break
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        return buf.decode("utf-8", errors="ignore")


def _parse_get_response(text: str) -> str | None:
    quoted = re.search(r"=\s*'([^']*)'", text)
    if quoted is not None:
        return quoted.group(1)
    plain = re.search(r"=\s*([^\r\n]+)", text)
    if plain is None:
        return None
    value = plain.group(1).strip()
    if " (" in value:
        value = value.split(" (", 1)[0].strip()
    return value if value else None


def get_property(
    path: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
) -> str | None:
    return _parse_get_response(_telnet_command(f"get {path}", port=port, timeout=timeout))


def set_property(
    path: str,
    value: str | int | float,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
) -> None:
    if isinstance(value, float):
        value_text = f"{value:.6f}"
    else:
        value_text = str(value)
    _telnet_command(f"set {path} {value_text}", port=port, timeout=timeout, expect_response=False)


def list_available_views(
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
    max_views: int = 32,
) -> list[tuple[int, str]]:
    views: list[tuple[int, str]] = []
    consecutive_misses = 0
    for idx in range(max_views):
        name = get_property(f"/sim/view[{idx}]/name", port=port, timeout=timeout)
        if name is None:
            consecutive_misses += 1
            if views and consecutive_misses >= 4:
                break
            continue
        views.append((idx, name))
        consecutive_misses = 0
    return views


def resolve_view_name(
    requested: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
) -> tuple[int, str]:
    views = list_available_views(port=port, timeout=timeout)
    if not views:
        raise RuntimeError("FlightGear reported no available views over telnet.")

    requested_norm = _normalize_view_name(requested)
    normalized_views = [(_normalize_view_name(name), idx, name) for idx, name in views]

    for norm_name, idx, name in normalized_views:
        if norm_name == requested_norm:
            return idx, name

    candidates = tuple(_normalize_view_name(name) for name in _VIEW_ALIAS_CANDIDATES.get(requested_norm, (requested,)))
    for candidate in candidates:
        for norm_name, idx, name in normalized_views:
            if norm_name == candidate:
                return idx, name

    for candidate in candidates:
        for norm_name, idx, name in normalized_views:
            if candidate in norm_name:
                return idx, name

    available = ", ".join(name for _, name in views)
    raise RuntimeError(f"Could not resolve FlightGear view {requested!r}. Available views: {available}")


def select_view(
    requested: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
) -> tuple[int, str]:
    idx, actual_name = resolve_view_name(requested, port=port, timeout=timeout)
    set_property("/sim/current-view/view-number", idx, port=port, timeout=timeout)
    return idx, actual_name


def configure_inspection_view(
    requested: str,
    *,
    cockpit_fov: float,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
    settle_seconds: float = 0.35,
) -> str:
    idx, actual_name = select_view(requested, port=port, timeout=timeout)
    time.sleep(settle_seconds)
    if _normalize_view_name(requested) == "cockpit":
        for path in (
            "/sim/current-view/field-of-view",
            "/sim/current-view/config/field-of-view",
            "/sim/current-view/config/default-field-of-view-deg",
            f"/sim/view[{idx}]/config/field-of-view",
            f"/sim/view[{idx}]/config/default-field-of-view-deg",
        ):
            set_property(path, cockpit_fov, port=port, timeout=timeout)
        time.sleep(settle_seconds)
    return actual_name


def _build_fg_args(aircraft: str) -> list[str]:
    return [
        f"--aircraft={aircraft}",
        "--graphics-preset=ultra-quality",
        f"--max-fps={_FG_MAX_FPS}",
        f"--texture-filtering={_FG_TEXTURE_FILTERING}",
        f"--native-fdm=socket,in,{_FG_NATIVE_FDM_RATE_HZ},,{_FG_FDM_PORT},udp",
        f"--telnet=socket,bi,60,,{_TELNET_PORT},tcp",
        "--fdm=null",
        "--in-air",
        "--altitude=10000",
        "--vc=200",
    ]


def wait_until_ready(
    port: int = _TELNET_PORT,
    timeout: float = 180.0,
    poll_interval: float = 1.0,
) -> float:
    """Poll FG's telnet-props port until /sim/sceneryloaded is true.

    Returns the elapsed seconds. Raises TimeoutError if FG never signals ready.
    The telnet socket only opens partway through FG startup, so early connection
    failures are expected — we keep retrying until the deadline.
    """
    deadline = time.monotonic() + timeout
    start = time.monotonic()
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
                s.settimeout(2.0)
                s.sendall(b"get /sim/sceneryloaded\r\n")
                buf = b""
                while b"\n" not in buf and len(buf) < 4096:
                    chunk = s.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                # Response format: "/sim/sceneryloaded = 'true' (BOOL)"
                low = buf.lower()
                if b"'true'" in low or b"= true" in low:
                    return time.monotonic() - start
        except (ConnectionRefusedError, OSError) as e:
            last_err = e
        time.sleep(poll_interval)
    raise TimeoutError(
        f"FlightGear not ready after {timeout:.0f}s "
        f"(telnet port {port}, last error: {last_err!r})"
    )


def find_user_aircraft_paths() -> list[Path]:
    """Find aircraft dirs the FG launcher knows about but ``fgfs.exe`` doesn't auto-discover.

    Standalone ``fgfs.exe`` only sees its bundled aircraft. The FG launcher tracks
    extra hangar download dirs (typically ``~/FlightGear/Downloads/Aircraft/<catalog-id>/Aircraft/``).
    We scan the standard user-download root and return every ``Aircraft/`` subdir
    we find so we can pass them via ``--fg-aircraft``.
    """
    paths: list[Path] = []
    candidates = [
        Path.home() / "FlightGear" / "Downloads" / "Aircraft",
        Path.home() / "Documents" / "FlightGear" / "Aircraft",
    ]
    for root in candidates:
        if not root.exists():
            continue
        # Direct aircraft (rare — usually nested under a catalog id)
        if any(p.is_dir() and p.name.endswith("-set.xml") is False for p in root.iterdir()):
            paths.append(root)
        # Catalog-nested layout: <root>/<catalog-id>/Aircraft/<package>/<aircraft-set.xml>
        for catalog_dir in root.iterdir():
            ac_dir = catalog_dir / "Aircraft"
            if ac_dir.is_dir():
                paths.append(ac_dir)
    # De-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_fgfs() -> Path | None:
    """Return the path to ``fgfs`` (or ``fgfs.exe``) if installed, else ``None``."""
    on_path = shutil.which("fgfs") or shutil.which("fgfs.exe")
    if on_path:
        return Path(on_path)

    if sys.platform == "win32":
        for pf in (r"C:\Program Files", r"C:\Program Files (x86)"):
            try:
                for candidate in sorted(Path(pf).glob("FlightGear*/bin/fgfs.exe"), reverse=True):
                    if candidate.exists():
                        return candidate
            except OSError:
                pass
    elif sys.platform.startswith("linux"):
        for p in ("/usr/games/fgfs", "/usr/bin/fgfs", "/usr/local/bin/fgfs",
                  "/opt/flightgear/bin/fgfs"):
            if Path(p).exists():
                return Path(p)
    elif sys.platform == "darwin":
        for p in ("/Applications/FlightGear.app/Contents/MacOS/fgfs",
                  "/Applications/FlightGear.app/Contents/Resources/fgfs"):
            if Path(p).exists():
                return Path(p)
    return None


def launch_flightgear(
    aircraft: str = _DEFAULT_AIRCRAFT,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Spawn FlightGear with our stall-recovery setup, detached.

    ``aircraft`` is the FG visual model (default ``c172p``; pass ``737-300``
    or similar if you've downloaded extra aircraft via the FG launcher).
    Flight dynamics still come from JSBSim's 737 via the native-FDM socket.

    Raises FileNotFoundError with a helpful install hint if ``fgfs`` isn't
    discoverable. The returned ``Popen`` keeps running after this Python
    process exits, so the FG window survives ``play``'s exit.
    """
    fgfs = find_fgfs()
    if fgfs is None:
        raise FileNotFoundError(
            "FlightGear (fgfs) not found. Install from https://www.flightgear.org/download/. "
            "On Windows we look under 'C:\\Program Files\\FlightGear*\\bin\\fgfs.exe'; "
            "on Linux at /usr/games/fgfs or /usr/local/bin/fgfs; "
            "on macOS at /Applications/FlightGear.app."
        )

    args = [str(fgfs)] + _build_fg_args(aircraft)
    user_paths = find_user_aircraft_paths()
    if user_paths:
        # FG accepts os-specific path separator (':' on Linux/Mac, ';' on Windows).
        joined = os.pathsep.join(str(p) for p in user_paths)
        args.append(f"--fg-aircraft={joined}")
    args += list(extra_args or [])
    popen_kwargs: dict = {
        # FG's Nasal scripts spam the parent terminal once per frame
        # (FMU Toast, navdisplay errors, etc.) — drop it all on the floor.
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin":  subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True

    return subprocess.Popen(args, **popen_kwargs)


def main() -> None:
    """``uv run flightgear`` entry point: just spawn FG with the defaults.

    Optional CLI: ``uv run flightgear --aircraft 737-300``.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--aircraft", default=_DEFAULT_AIRCRAFT,
                   help="FG visual model. Default 737-300. Must be installed via "
                        "the FG launcher first - see README.")
    args = p.parse_args()
    proc = launch_flightgear(aircraft=args.aircraft)
    print(f"FlightGear PID {proc.pid} - cockpit window will appear in ~10-15 s.")
    print(f"(Visual model: {args.aircraft}. Dynamics come from JSBSim's 737.)")
    print("The FlightGear window stays up after this command exits.")


if __name__ == "__main__":
    main()
