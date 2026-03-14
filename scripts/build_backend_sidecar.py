from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _host_target_triple() -> str:
    out = subprocess.check_output(["rustc", "-vV"], text=True)
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Failed to read rust host target triple.")


def main() -> None:
    root = _repo_root()
    entry = root / "desktop_backend_entry.py"
    if not entry.exists():
        raise FileNotFoundError(f"Missing backend sidecar entry: {entry}")

    target = _host_target_triple()
    is_windows = "windows" in target
    sidecar_base = "mcube-backend"
    pyinstaller_name = sidecar_base

    dist_dir = root / ".tmp_sidecar_dist"
    work_dir = root / ".tmp_sidecar_build"
    spec_dir = root / ".tmp_sidecar_spec"
    for d in (dist_dir, work_dir, spec_dir):
        d.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        pyinstaller_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        str(entry),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    built_dir = dist_dir / pyinstaller_name
    built_exe = built_dir / (f"{sidecar_base}.exe" if is_windows else sidecar_base)
    if not built_exe.exists():
        raise FileNotFoundError(f"Expected built sidecar executable not found: {built_exe}")

    binaries_dir = root / "frontend" / "src-tauri" / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)
    # Clean previously copied onedir files to avoid stale runtime deps.
    for item in binaries_dir.iterdir():
        if item.name == ".gitkeep":
            continue
        if item.is_file():
            item.unlink(missing_ok=True)
        else:
            shutil.rmtree(item, ignore_errors=True)

    tauri_sidecar_name = f"{sidecar_base}-{target}{'.exe' if is_windows else ''}"
    for item in built_dir.iterdir():
        if item.is_file():
            out_name = tauri_sidecar_name if item.name == built_exe.name else item.name
            shutil.copy2(item, binaries_dir / out_name)
        elif item.is_dir():
            shutil.copytree(item, binaries_dir / item.name, dirs_exist_ok=True)

    # Mark executable bit on non-Windows.
    if not is_windows:
        out_file = binaries_dir / tauri_sidecar_name
        out_file.chmod(out_file.stat().st_mode | 0o111)

    print(f"Sidecar ready (onedir): {binaries_dir}")


if __name__ == "__main__":
    main()
