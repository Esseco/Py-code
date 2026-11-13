import csv
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from Process_MLIP.direct_sampling.pca import structures_to_pca_csv


def make_atoms(scale):
    return Atoms("Li2", positions=[[0, 0, 0], [1.5, 0, 0]], cell=[4 + scale, 4, 4], pbc=True)


def stub(atoms):
    return np.array([atoms.cell[0, 0], atoms.get_volume(), float(len(atoms))])


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def test_one_row_per_structure_with_two_components(tmp_path):
    structures = [make_atoms(i * 0.1) for i in range(6)]
    report = structures_to_pca_csv(structures, tmp_path / "pca.csv", extractor=stub)
    data = rows(tmp_path / "pca.csv")
    assert len(data) == 6
    assert list(data[0]) == ["structure_index", "source", "frame_index", "label", "pc1", "pc2"]
    assert report["n_components"] == 2 and report["feature_dim"] == 3
    assert report["explained_variance_ratio"][0] >= report["explained_variance_ratio"][1]
    assert [r["source"] for r in data] == ["<memory>"] * 6
    assert all(np.isfinite(float(r["pc1"]) + float(r["pc2"])) for r in data)


def test_trajectory_path_expands_frames_and_keeps_labels(tmp_path):
    trajectory = tmp_path / "frames.extxyz"
    write(trajectory, [make_atoms(i * 0.1) for i in range(4)])
    report = structures_to_pca_csv([str(trajectory)], tmp_path / "pca.csv", extractor=stub, labels=list("abcd"))
    data = rows(tmp_path / "pca.csv")
    assert [int(r["frame_index"]) for r in data] == [0, 1, 2, 3]
    assert [r["label"] for r in data] == list("abcd")
    assert {r["source"] for r in data} == {str(trajectory.resolve())}
    assert report["n_structures"] == 4


def test_guards(tmp_path):
    structures = [make_atoms(i * 0.1) for i in range(3)]
    output = tmp_path / "pca.csv"
    structures_to_pca_csv(structures, output, extractor=stub)
    with pytest.raises(FileExistsError):
        structures_to_pca_csv(structures, output, extractor=stub)
    structures_to_pca_csv(structures, output, extractor=stub, overwrite=True)
    with pytest.raises(ValueError, match="at least two structures"):
        structures_to_pca_csv(structures[:1], tmp_path / "one.csv", extractor=stub)
    with pytest.raises(ValueError, match="exceeds the usable rank"):
        structures_to_pca_csv(structures, tmp_path / "rank.csv", extractor=stub, n_components=4)
    with pytest.raises(ValueError, match="one entry per structure"):
        structures_to_pca_csv(structures, tmp_path / "label.csv", extractor=stub, labels=["a"])
    with pytest.raises(ValueError, match="model="):
        structures_to_pca_csv(structures, tmp_path / "nomodel.csv")


def test_inconsistent_feature_dimension_is_rejected(tmp_path):
    def uneven(atoms):
        return np.zeros(3 if atoms.cell[0, 0] < 4.15 else 5)

    with pytest.raises(ValueError, match="inconsistent feature dimension"):
        structures_to_pca_csv([make_atoms(0.0), make_atoms(1.0)], tmp_path / "pca.csv", extractor=uneven)


def test_cli_pca_subcommand_is_registered():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "Process_MLIP.direct_sampling", "pca", "--help"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout and "structures" in result.stdout

@pytest.mark.skipif(not os.environ.get("M3GNET_CHECKPOINT"), reason="set M3GNET_CHECKPOINT to a checkpoint directory to run the real encoder")
def test_real_m3gnet_checkpoint_produces_separable_csv(tmp_path):
    pytest.importorskip("tensorflow")
    from ase.build import bulk

    from Process_MLIP.direct_sampling.features import build_encoder

    encoder = build_encoder(os.environ["M3GNET_CHECKPOINT"])
    frames = [bulk("Li", "bcc", a=a, cubic=True) for a in (3.2, 3.4, 3.6)]
    vectors = [encoder(atoms) for atoms in frames]
    assert all(v.ndim == 1 and v.size > 1 for v in vectors), [v.shape for v in vectors]
    assert all(np.isfinite(v).all() for v in vectors)
    report = structures_to_pca_csv(frames, tmp_path / "real.csv", extractor=encoder)
    data = rows(tmp_path / "real.csv")
    assert len(data) == 3 and report["feature_dim"] == vectors[0].size
    assert len({r["pc1"] for r in data}) == 3
