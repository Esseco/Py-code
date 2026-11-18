"""Run the unified extractor with three different feature selections."""

from pathlib import Path

from pymatgen.core import Lattice, Species, Structure

from Process_Struct.features import FeatureExtractor, available_sets, extract_features

MODEL_PATH = Path("E:/Py-Code/MLIP_model/MACE")
ENCODER = "mace"


def rocksalt_nacl() -> Structure:
    """Cubic NaCl with oxidation states, so CrystalNN resolves six neighbours."""

    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        [Species("Na", 1), Species("Cl", -1)],
        [[0, 0, 0], [0.5, 0, 0]],
    )


def main() -> None:
    structure = rocksalt_nacl()

    print("available sets:")
    for name, info in available_sets().items():
        print(f"  {name:9s} {info['prefix']:9s} {info['levels']} {info['description']}")

    print("\n[1] geometry only (default, no extra dependencies)")
    geometry = extract_features(structure)
    print(f"  {len(geometry.names)} columns, e.g. {list(geometry.values)[:3]}")

    print("\n[2] geometry + CrystalNN coordination statistics, per site")
    both = FeatureExtractor(sets=["geometry", "cnn"], level="both")(structure)
    print(f"  structure columns: {len(both.names)}, sites: {len(both.sites)}")
    print(f"  cn_mean = {both.values['cnn.cn_mean']}, first site: {both.sites[0]}")

    print("\n[3] optional sets need their own environment")
    try:
        fingerprints = extract_features(structure, sets=["cnn_fp"])
        print(f"  cnn_fp columns: {len(fingerprints.names)}")
    except ImportError as error:
        print(f"  cnn_fp unavailable here: {error}")
    print(f"  mlip example: FeatureExtractor(sets=['mlip'], encoder={ENCODER!r}, "
          f"model_path={str(MODEL_PATH)!r})")
    print("  (loads a real model once, then call the extractor per structure)")


if __name__ == "__main__":
    main()
