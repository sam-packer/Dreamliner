"""FlightGear discovery + launching.

The Windows installer doesn't add ``fgfs.exe`` to ``PATH``, so we look in the
standard install locations. ``launch_flightgear()`` spawns it detached so the
cockpit window survives even after ``play`` exits.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

# Default visual: Boeing 737-300. Download it once via the FlightGear launcher
# (Aircraft tab -> search "737-300" -> Install) before running. JSBSim drives
# the actual flight dynamics through the native-FDM socket; the FG aircraft is
# the visual / cockpit only.
_DEFAULT_AIRCRAFT = "737-300"

# FG's telnet-props interface. We query /sim/sceneryloaded to know when the
# cockpit is actually ready to receive FDM packets (scenery load is the slow
# step — often 30-60s on first launch).
_TELNET_PORT = 5501
_TELNET_TIMEOUT_SECS = 2.0
_TIME_OF_DAY_LOCAL_MINUTES: dict[str, int] = {
    "real": -1,
    "dawn": 5 * 60 + 30,
    "morning": 7 * 60,
    "noon": 12 * 60,
    "afternoon": 15 * 60 + 20,
    "dusk": 18 * 60 + 40,
    "evening": 19 * 60 + 50,
    "midnight": 0,
}
_VIEW_ALIAS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "cockpit": ("Pilot View", "Captain View", "Captain", "Cockpit View", "Pilot"),
    "chase": ("Chase View", "Helicopter View", "Chase"),
    "tower": ("Tower View", "Tower"),
    "fly-by": ("Fly-By View", "Fly By View", "Fly-by"),
}


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


def _get_float_property(
    path: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
) -> float | None:
    raw = get_property(path, port=port, timeout=timeout)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _set_daytime_gmt(
    preset: str,
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
    invert_offset: bool = False,
) -> None:
    current_gmt = get_property("/sim/time/gmt", port=port, timeout=timeout)
    if current_gmt is None:
        return
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}$", current_gmt)
    if match is None:
        return
    target_local_minutes = _TIME_OF_DAY_LOCAL_MINUTES[preset]
    local_offset_seconds = _get_float_property("/sim/time/local-offset", port=port, timeout=timeout) or 0.0
    offset_seconds = -local_offset_seconds if invert_offset else local_offset_seconds
    target_utc_seconds = int((target_local_minutes * 60 - offset_seconds) % (24 * 60 * 60))
    hour = target_utc_seconds // 3600
    minute = (target_utc_seconds % 3600) // 60
    second = target_utc_seconds % 60
    year, month, day = match.groups()
    set_property(
        "/sim/time/gmt",
        f"{year}-{month}-{day}T{hour:02d}:{minute:02d}:{second:02d}",
        port=port,
        timeout=timeout,
    )


def _parse_clock_minutes(text: str | None) -> int | None:
    if text is None:
        return None
    match = re.match(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return hour * 60 + minute


def force_time_of_day(
    preset: str = "afternoon",
    *,
    port: int = _TELNET_PORT,
    timeout: float = _TELNET_TIMEOUT_SECS,
    settle_seconds: float = 0.2,
    attempts: int = 3,
) -> None:
    if preset not in _TIME_OF_DAY_LOCAL_MINUTES:
        raise ValueError(f"Unsupported FlightGear time-of-day preset: {preset}")
    if preset == "real":
        return
    for _ in range(attempts):
        set_property("/sim/multiplay/lag/master", 0, port=port, timeout=timeout)
        set_property("/sim/time/simple-time/enabled", 1, port=port, timeout=timeout)
        set_property("/sim/time/warp-delta", 0, port=port, timeout=timeout)
        set_property("/sim/time/warp-easing", 0, port=port, timeout=timeout)
        _set_daytime_gmt(preset, port=port, timeout=timeout)
        time.sleep(settle_seconds)
        local_minutes = _parse_clock_minutes(get_property("/sim/time/local-time-string", port=port, timeout=timeout))
        target_local_minutes = _TIME_OF_DAY_LOCAL_MINUTES[preset]
        if local_minutes is not None:
            delta = min(
                (local_minutes - target_local_minutes) % (24 * 60),
                (target_local_minutes - local_minutes) % (24 * 60),
            )
            if delta > 45:
                _set_daytime_gmt(preset, port=port, timeout=timeout, invert_offset=True)
                time.sleep(settle_seconds)


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
        "--ignore-autosave",
        "--timeofday=afternoon",
        "--native-fdm=socket,in,60,,5550,udp",
        f"--telnet=socket,bi,60,,{_TELNET_PORT},tcp",
        "--fdm=external",
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
