from __future__ import annotations

import os
from pathlib import Path

_PIPELINE_ENV_KEYS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "BUILDKITE",
    "TF_BUILD",
    "JENKINS_URL",
)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_pipeline_environment() -> bool:
    return any(_is_truthy(os.environ.get(key)) for key in _PIPELINE_ENV_KEYS)


def _load_dotenv_into_environ(env_path: Path) -> None:
    if not env_path.exists() or _is_pipeline_environment():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_load_dotenv_into_environ(_REPO_ROOT / ".env")
