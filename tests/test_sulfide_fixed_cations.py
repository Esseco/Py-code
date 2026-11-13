import pytest

from Process_Struct.Chemical_Capacity_Constraints import estimate_redox_capacity


@pytest.mark.parametrize(
    "formula",
    [
        "Li2S",
        "MgS",
        "Al2S3",
        "ZnS",
        "ZrS2",
        "Ga2S3",
        "GeS2",
        "CdS",
        "PbS",
        "Li2FeSe2",
        "Li2FeTe2",
    ],
)
def test_sulfide_recognizes_fixed_valence_cations_and_chalcogen_anions(formula):
    result = estimate_redox_capacity(formula, environment="auto")

    assert result["environment"] == "sulfide"
    assert result["status"] in {"ok", "no_redox_active_elements"}
    assert result["unresolved_elements"] == []
