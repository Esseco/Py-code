"""Select and guard electerm HPC connection profiles.

Call ``preflight_hpc`` before every BJ-HPC electerm operation and pass its
receipt to ``require_preflight`` immediately before the MCP call. The receipt
binds the operation to the selected profile, host, tab, and exact skill text.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _ROOT / "tools" / "electerm_hpc_profiles.json"
DEFAULT_PREFLIGHT_TTL_SECONDS = 30 * 60


class ConfigurationError(ValueError):
    """Raised when an HPC profile cannot be safely used."""


class PreflightRequired(ConfigurationError):
    """Raised when an electerm call has no valid current preflight receipt."""


@dataclass(frozen=True)
class HPCContext:
    """Resolved profile and the skill text that must govern the connection."""

    name: str
    display_name: str
    host: str
    skill_path: Path
    skill_text: str
    repository: str | None
    task_root: str | None
    environment: str | None


@dataclass(frozen=True)
class PreflightReceipt:
    """Short-lived proof that a specific HPC/tab passed the local preflight."""

    profile: str
    host: str
    skill_path: Path
    skill_sha256: str
    issued_at: float
    expires_at: float
    tab_id: str | None = None


def _skill_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_path(value: str, *, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else (base / path).resolve()


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read HPC profile config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in HPC profile config: {config_path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        raise ConfigurationError("HPC profile config must contain an object named 'profiles'")
    return data


def select_hpc(
    name: str | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    connected_host: str | None = None,
) -> HPCContext:
    """Resolve an HPC profile, read its bound skill, and optionally verify host."""

    config_file = Path(config_path).expanduser().resolve()
    data = _load_config(config_file)
    profile_name = name or data.get("default_profile")
    if not isinstance(profile_name, str) or not profile_name:
        raise ConfigurationError("no HPC profile was selected and no default_profile is configured")
    raw = data["profiles"].get(profile_name)
    if not isinstance(raw, dict):
        available = ", ".join(sorted(data["profiles"]))
        raise ConfigurationError(f"unknown HPC profile '{profile_name}'; available: {available}")

    host = raw.get("host")
    skill = raw.get("skill")
    if not isinstance(host, str) or not host.strip():
        raise ConfigurationError(f"profile '{profile_name}' has no host")
    if not isinstance(skill, str) or not skill.strip():
        raise ConfigurationError(f"profile '{profile_name}' has no bound skill file")
    skill_path = _resolve_path(skill, base=config_file.parent)
    try:
        skill_text = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"profile '{profile_name}' skill file is missing or unreadable: {skill_path}"
        ) from exc
    if not skill_text.strip():
        raise ConfigurationError(f"profile '{profile_name}' skill file is empty: {skill_path}")
    if connected_host is not None and connected_host.strip().lower() != host.strip().lower():
        raise ConfigurationError(
            f"connected host '{connected_host}' does not match profile '{profile_name}' host '{host}'"
        )

    return HPCContext(
        name=profile_name,
        display_name=str(raw.get("display_name", profile_name)),
        host=host,
        skill_path=skill_path,
        skill_text=skill_text,
        repository=raw.get("repository"),
        task_root=raw.get("task_root"),
        environment=raw.get("environment"),
    )


def preflight_hpc(
    name: str | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    connected_host: str,
    tab_id: str | None = None,
    ttl_seconds: float = DEFAULT_PREFLIGHT_TTL_SECONDS,
) -> PreflightReceipt:
    """Run the mandatory preflight and return a short-lived call receipt."""

    if not connected_host.strip():
        raise ConfigurationError("preflight requires the expected connected host")
    if ttl_seconds <= 0:
        raise ConfigurationError("preflight ttl_seconds must be positive")
    context = select_hpc(name, config_path=config_path, connected_host=connected_host)
    issued_at = time.time()
    return PreflightReceipt(
        profile=context.name,
        host=context.host,
        skill_path=context.skill_path,
        skill_sha256=_skill_sha256(context.skill_text),
        issued_at=issued_at,
        expires_at=issued_at + ttl_seconds,
        tab_id=tab_id,
    )


def require_preflight(
    receipt: PreflightReceipt | None,
    *,
    profile: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    connected_host: str,
    tab_id: str | None,
) -> HPCContext:
    """Reject an electerm operation unless its current receipt still matches."""

    if not isinstance(receipt, PreflightReceipt):
        raise PreflightRequired("electerm operation rejected: preflight is required")
    now = time.time()
    if now >= receipt.expires_at:
        raise PreflightRequired("electerm operation rejected: preflight has expired")
    if receipt.profile != profile:
        raise PreflightRequired("electerm operation rejected: profile changed")
    if receipt.host.strip().lower() != connected_host.strip().lower():
        raise PreflightRequired("electerm operation rejected: host changed")
    if receipt.tab_id != tab_id:
        raise PreflightRequired("electerm operation rejected: tab changed or was not bound")

    context = select_hpc(profile, config_path=config_path, connected_host=connected_host)
    if context.skill_path != receipt.skill_path:
        raise PreflightRequired("electerm operation rejected: skill path changed")
    if _skill_sha256(context.skill_text) != receipt.skill_sha256:
        raise PreflightRequired("electerm operation rejected: skill content changed")
    return context


def main() -> int:
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="预检 electerm HPC 配置并读取绑定 skill")
    parser.add_argument("--profile", help="HPC 档案名称；省略时使用 default_profile")
    parser.add_argument("--host", dest="connected_host", required=True, help="实际 electerm 连接主机")
    parser.add_argument("--tab-id", help="已连接的 electerm tab ID")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="档案 JSON 路径")
    args = parser.parse_args()
    try:
        receipt = preflight_hpc(
            args.profile,
            config_path=args.config,
            connected_host=args.connected_host,
            tab_id=args.tab_id,
        )
        context = select_hpc(args.profile, config_path=args.config, connected_host=args.connected_host)
    except ConfigurationError as exc:
        parser.error(str(exc))
    print(_json.dumps({
        "profile": context.name,
        "display_name": context.display_name,
        "host": context.host,
        "skill_path": str(context.skill_path),
        "skill_sha256": receipt.skill_sha256,
        "skill_loaded": True,
        "preflight_expires_at": receipt.expires_at,
        "tab_id": receipt.tab_id,
        "repository": context.repository,
        "task_root": context.task_root,
        "environment": context.environment,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
