"""Per-element aggregation of the final OUTCAR moments; no VASP execution."""
from unittest.mock import patch

import pytest

from Process_Vasp import read_element_moments

COLLINEAR = {
    "status": "available",
    "noncollinear": False,
    "source": "OUTCAR_final_local_projection",
    "elements": ["Fe", "O", "Fe"],
    "total_local_moment": 8.31,
    "sites": [
        {"atom_index": 0, "element": "Fe", "moment": 4.3},
        {"atom_index": 1, "element": "O", "moment": 0.01},
        {"atom_index": 2, "element": "Fe", "moment": 4.0},
    ],
}

NONCOLLINEAR = {
    "status": "available",
    "noncollinear": True,
    "source": "OUTCAR_final_local_projection",
    "elements": ["Mn", "O"],
    "total_local_moment": None,
    "sites": [
        {"atom_index": 0, "element": "Mn", "moment": [0.0, 0.0, 4.0]},
        {"atom_index": 1, "element": "O", "moment": [0.0, 0.0, -0.5]},
    ],
}


def _read(raw, tmp_path):
    with patch(
        "Process_Vasp.results.magnetic_check.read_dft_magnetic_data", return_value=raw
    ):
        return read_element_moments(tmp_path)


def test_collinear_aggregation_keeps_atom_order(tmp_path):
    report = _read(COLLINEAR, tmp_path)
    assert report["available"] is True
    assert report["noncollinear"] is False
    assert list(report["elements"]) == ["Fe", "O"]
    assert report["atom_elements"] == ["Fe", "O", "Fe"]
    fe = report["elements"]["Fe"]
    assert fe["count"] == 2
    assert fe["sum"] == pytest.approx(8.3)
    assert fe["mean"] == pytest.approx(4.15)
    assert (fe["min"], fe["max"]) == (4.0, 4.3)
    assert fe["moments"] == [4.3, 4.0]
    assert report["total_local_moment"] == pytest.approx(8.31)
    assert report["n_atoms"] == 3


def test_noncollinear_keeps_vectors_and_norm(tmp_path):
    report = _read(NONCOLLINEAR, tmp_path)
    assert report["noncollinear"] is True
    mn = report["elements"]["Mn"]
    assert mn["mean"] == [0.0, 0.0, 4.0]
    assert mn["norm_mean"] == pytest.approx(4.0)
    assert "min" not in mn and "max" not in mn


def test_missing_data_is_explicit(tmp_path):
    unavailable = {
        "status": "unavailable",
        "noncollinear": False,
        "read_error": "FileNotFoundError: OUTCAR",
    }
    report = _read(unavailable, tmp_path)
    assert report["available"] is False
    assert report["status"] == "unavailable"
    assert "FileNotFoundError" in report["error"]
    assert report["elements"] == {}
