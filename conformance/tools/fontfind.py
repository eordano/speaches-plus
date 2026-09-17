import os
import subprocess


def font_file(basename, env=None):
    if env and os.environ.get(env):
        return os.environ[env]
    try:
        listed = subprocess.run(
            ["fc-list", "-f", "%{file}\n"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        listed = ""
    hits = sorted(p for p in listed.splitlines() if p.endswith("/" + basename))
    if hits:
        return hits[0]
    raise SystemExit(
        f"{basename} not found ({env or 'no'} env override set); "
        "run conformance/tools/regenerate-fixtures.sh, which locates fonts and names the nix shell"
    )
