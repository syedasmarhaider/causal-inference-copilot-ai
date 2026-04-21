from __future__ import annotations

import math
import os
from pathlib import Path

import uvicorn

APP_IMPORT_PATH = "python.adapters.api.app:app"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080


def _read_positive_int_env(name: str) -> int | None:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= 1") from exc

    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _resolve_host() -> str:
    return (
        os.getenv("HOST")
        or os.getenv("API_HOST")
        or DEFAULT_HOST
    )


def _resolve_port() -> int:
    return (
        _read_positive_int_env("PORT")
        or _read_positive_int_env("API_PORT")
        or DEFAULT_PORT
    )


def _resolve_worker_count() -> int:
    configured_workers = _read_positive_int_env("WEB_CONCURRENCY")
    if configured_workers is not None:
        return configured_workers
    return _detect_available_cpus()


def _detect_available_cpus() -> int:
    for detector in (
        _cpu_count_from_affinity,
        _cpu_count_from_cgroup_v2,
        _cpu_count_from_cgroup_v1,
        os.cpu_count,
    ):
        detected = detector()
        if detected and detected > 0:
            return detected
    return 1


def _cpu_count_from_affinity() -> int | None:
    if not hasattr(os, "sched_getaffinity"):
        return None

    try:
        return len(os.sched_getaffinity(0))
    except OSError:
        return None


def _cpu_count_from_cgroup_v2() -> int | None:
    cpu_max = _read_text_file(Path("/sys/fs/cgroup/cpu.max"))
    if cpu_max is None:
        return None

    quota, _, period = cpu_max.partition(" ")
    if quota == "max":
        return None
    return _cpu_count_from_quota(quota=quota, period=period)


def _cpu_count_from_cgroup_v1() -> int | None:
    quota = _read_text_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = _read_text_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota is None or period is None:
        return None
    return _cpu_count_from_quota(quota=quota, period=period)


def _cpu_count_from_quota(*, quota: str, period: str) -> int | None:
    try:
        quota_value = int(quota)
        period_value = int(period)
    except ValueError:
        return None

    if quota_value <= 0 or period_value <= 0:
        return None

    return max(1, math.ceil(quota_value / period_value))


def _read_text_file(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def main() -> None:
    uvicorn.run(
        APP_IMPORT_PATH,
        host=_resolve_host(),
        port=_resolve_port(),
        workers=_resolve_worker_count(),
    )


if __name__ == "__main__":
    main()
