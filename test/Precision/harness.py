"""Shared plumbing for the precision sweeps: locating the interpreter, a
scratch directory, and running one generated kernel."""
import os
import subprocess
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))


def interpreter():
    """Path to npuir-interp.

    `NPUIR_INTERP` wins; otherwise look for the in-tree build, which is
    where it lands for anyone running this from a source checkout.
    """
    env = os.environ.get("NPUIR_INTERP")
    if env:
        return env
    root = os.path.abspath(os.path.join(_HERE, "..", ".."))
    guess = os.path.join(root, "build", "bin", "npuir-interp")
    if os.path.exists(guess):
        return guess
    raise SystemExit(
        "cannot find npuir-interp; set NPUIR_INTERP to its path")


def scratch(name):
    """A per-sweep scratch directory. `NPUIR_INTERP_SCRATCH` overrides."""
    base = os.environ.get("NPUIR_INTERP_SCRATCH") or tempfile.gettempdir()
    path = os.path.join(base, "npuir-interp-precision", name)
    os.makedirs(path, exist_ok=True)
    return path


def run(src, args, out_prefix, extra=()):
    """Run one kernel. Returns (returncode, combined output)."""
    cmd = [interpreter(), src, "--sched=lazy", *extra,
           "--args=" + ",".join(args), "--out=" + out_prefix]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr
