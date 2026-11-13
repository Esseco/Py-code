from pathlib import Path
import json
import pytest

from tools.electerm_hpc import (
    ConfigurationError,
    PreflightReceipt,
    PreflightRequired,
    preflight_hpc,
    require_preflight,
    select_hpc,
)


def write_config(tmp_path: Path, *, skill_path: str, host: str = "hpc.example") -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({
        "default_profile": "test-hpc",
        "profiles": {
            "test-hpc": {
                "display_name": "Test HPC",
                "host": host,
                "skill": skill_path,
                "repository": "/repo",
                "task_root": "/tasks",
                "environment": "py1",
            }
        },
    }), encoding="utf-8")
    return path


def test_select_hpc_reads_bound_skill_and_validates_host(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("# Test HPC rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill))

    context = select_hpc("test-hpc", config_path=config, connected_host="hpc.example")

    assert context.name == "test-hpc"
    assert context.display_name == "Test HPC"
    assert context.skill_path == skill.resolve()
    assert context.skill_text == "# Test HPC rules"
    assert context.repository == "/repo"


def test_select_hpc_rejects_host_mismatch(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill), host="expected.example")

    with pytest.raises(ConfigurationError, match="does not match"):
        select_hpc("test-hpc", config_path=config, connected_host="other.example")


def test_select_hpc_rejects_missing_skill(tmp_path):
    config = write_config(tmp_path, skill_path=str(tmp_path / "missing.md"))

    with pytest.raises(ConfigurationError, match="skill file"):
        select_hpc("test-hpc", config_path=config)


def test_select_hpc_uses_configured_default(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill))

    context = select_hpc(config_path=config)

    assert context.name == "test-hpc"


def test_preflight_receipt_binds_profile_host_skill_and_tab(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill))

    receipt = preflight_hpc(
        "test-hpc", config_path=config, connected_host="hpc.example", tab_id="tab-1"
    )

    assert isinstance(receipt, PreflightReceipt)
    assert receipt.skill_sha256
    context = require_preflight(
        receipt,
        profile="test-hpc",
        config_path=config,
        connected_host="hpc.example",
        tab_id="tab-1",
    )
    assert context.name == "test-hpc"


def test_require_preflight_rejects_missing_or_mismatched_receipt(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill))

    with pytest.raises(PreflightRequired, match="preflight"):
        require_preflight(
            None,
            profile="test-hpc",
            config_path=config,
            connected_host="hpc.example",
            tab_id="tab-1",
        )

    receipt = preflight_hpc(
        "test-hpc", config_path=config, connected_host="hpc.example", tab_id="tab-1"
    )
    with pytest.raises(PreflightRequired, match="tab"):
        require_preflight(
            receipt,
            profile="test-hpc",
            config_path=config,
            connected_host="hpc.example",
            tab_id="tab-2",
        )


def test_require_preflight_rejects_changed_skill(tmp_path):
    skill = tmp_path / "test-hpc.md"
    skill.write_text("rules", encoding="utf-8")
    config = write_config(tmp_path, skill_path=str(skill))
    receipt = preflight_hpc(
        "test-hpc", config_path=config, connected_host="hpc.example", tab_id="tab-1"
    )
    skill.write_text("changed rules", encoding="utf-8")

    with pytest.raises(PreflightRequired, match="changed"):
        require_preflight(
            receipt,
            profile="test-hpc",
            config_path=config,
            connected_host="hpc.example",
            tab_id="tab-1",
        )
