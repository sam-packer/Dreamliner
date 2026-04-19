"""FlightGear discovery + launching.

The Windows installer doesn't add ``fgfs.exe`` to ``PATH``, so we look in the
standard install locations. ``launch_flightgear()`` spawns it detached so the
cockpit window survives even after ``play`` exits.
"""

from __future__ import annotations

import os
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


def _build_fg_args(aircraft: str) -> list[str]:
    return [
        f"--aircraft={aircraft}",
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
