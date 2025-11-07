#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def read_version_from_pyproject(path: str) -> str:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    proj = data.get("project", {})
    v = proj.get("version")
    if not v:
        raise SystemExit("Could not read version from pyproject")
    return str(v)


def parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    nums = []
    for p in parts[:3]:
        n = ""
        for ch in p:
            if ch.isdigit():
                n += ch
            else:
                break
        nums.append(int(n or 0))
    while len(nums) < 3:
        nums.append(0)
    return nums[0], nums[1], nums[2]


def get_prev_pypi_version(pkg: str, current: str | None) -> str | None:
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{pkg}/json", timeout=10
        ) as r:
            meta = json.load(r)
    except Exception:
        return None
    releases = list(meta.get("releases", {}).keys())
    if not releases:
        return None

    def key(s: str):
        return parse_version(s)

    if current is None:
        return sorted(releases, key=key, reverse=True)[0]
    cur_t = parse_version(current)
    older = [rv for rv in releases if parse_version(rv) < cur_t]
    if not older:
        return None
    return sorted(older, key=key, reverse=True)[0]


def generate_openapi_current(repo_root: Path, out_path: Path) -> None:
    # Ensure we can import the local package
    sys.path.insert(0, str(repo_root / "openhands-agent-server"))
    from openhands.agent_server.openapi import generate_openapi_schema  # type: ignore

    schema = generate_openapi_schema()
    out_path.write_text(json.dumps(schema, indent=2))


def generate_openapi_from_installed(version: str, out_path: Path) -> None:
    # Create a temporary virtualenv to avoid polluting runner env, using uv
    venv_dir = Path(tempfile.mkdtemp(prefix="old_agent_server_venv_"))
    subprocess.run(["uv", "venv", str(venv_dir)], check=True)
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    pybin = venv_dir / bin_dir / ("python.exe" if os.name == "nt" else "python")

    # Install the specific previous agent-server using uv
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(pybin),
            f"openhands-agent-server=={version}",
        ],
        check=True,
    )

    code = (
        "import json;"
        "from openhands.agent_server.api import api;"
        "schema = api.openapi();"
        f"open(r'{out_path.as_posix()}', 'w').write(json.dumps(schema, indent=2))"
    )
    subprocess.run([str(pybin), "-c", code], check=True)


def run_oasdiff(old_path: Path, new_path: Path) -> int:
    # Use oasdiff docker image with config file if present
    workdir = old_path.parent
    config_path = workdir / "oasdiff.yaml"
    docker_args = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workdir.as_posix()}:/work",
        "ghcr.io/tufin/oasdiff:latest",
        "breaking",
        f"/work/{old_path.name}",
        f"/work/{new_path.name}",
    ]
    if config_path.exists():
        # oasdiff discovers config file in CWD named oasdiff.*; set working directory
        docker_args.insert(6, "-w")
        docker_args.insert(7, "/work")
    # Ensure we fail CI on ERR-level by default (also in config)
    docker_args += ["--fail-on", "ERR", "-f", "githubactions"]
    proc = subprocess.run(docker_args)
    return proc.returncode


def main() -> int:
    repo_root = Path(os.getcwd())
    pyproj = repo_root / "openhands-agent-server" / "pyproject.toml"
    new_version = read_version_from_pyproject(str(pyproj))

    prev = get_prev_pypi_version("openhands-agent-server", new_version)
    if not prev:
        print(
            "::warning title=REST API::No previous openhands-agent-server release "
            "found; skipping breakage check"
        )
        return 0

    out_dir = repo_root / ".github" / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure oasdiff config is available in the working dir for the container
    config_src = repo_root / ".github" / "oasdiff.yaml"
    if config_src.exists():
        try:
            import shutil

            shutil.copyfile(config_src, out_dir / "oasdiff.yaml")
        except Exception as e:
            print(f"::warning title=REST API::Failed to copy oasdiff config: {e}")

    old_path = out_dir / "openapi-old.json"
    new_path = out_dir / "openapi-new.json"

    try:
        generate_openapi_current(repo_root, new_path)
    except Exception as e:
        print(f"::error title=REST API::Failed to generate current OpenAPI: {e}")
        return 1

    try:
        generate_openapi_from_installed(prev, old_path)
    except Exception as e:
        print(
            f"::error title=REST API::Failed to generate previous OpenAPI from "
            f"PyPI {prev}: {e}"
        )
        return 1

    code = run_oasdiff(old_path, new_path)
    if code == 0:
        print("No REST breaking changes detected")
        return 0

    # Non-zero means breaking changes detected or error; assume breaking for enforcement
    old_major, old_minor, _ = parse_version(prev)
    new_major, new_minor, _ = parse_version(new_version)

    ok = new_major > old_major  # Require major bump for REST
    if not ok:
        print(
            f"::error title=REST SemVer::Breaking REST changes detected; require "
            f"major bump from {old_major}.{old_minor}.x, but new is {new_version}"
        )
        return 1

    print("REST breaking changes detected and major bump policy satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
