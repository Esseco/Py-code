"""Check the unified feature entry point and its selection rules."""

import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Species, Structure
from pymatgen.io.vasp import Poscar

from Process_Struct import extract_features as top_level_extract
from Process_Struct.features import (
    FeatureExtractor,
    FeatureResult,
    available_sets,
    extract_features,
    extract_many,
    load_structure,
)


def rocksalt_nacl() -> Structure:
    """Eight-site NaCl with oxidation states so CrystalNN finds six neighbours."""

    return Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.64),
        [Species("Na", 1), Species("Cl", -1)],
        [[0, 0, 0], [0.5, 0, 0]],
    )


def layered_licoo2() -> Structure:
    return Structure.from_spacegroup(
        "R-3m",
        Lattice.hexagonal(2.82, 14.05),
        [Species("Li", 1), Species("Co", 3), Species("O", -2)],
        [[0, 0, 0.5], [0, 0, 0], [0, 0, 0.239]],
    )


class FeatureExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nacl = rocksalt_nacl()
        cls.licoo2 = layered_licoo2()

    def test_top_level_export_is_the_same_function(self):
        self.assertIs(top_level_extract, extract_features)

    def test_default_sets_are_geometry_only(self):
        result = extract_features(self.nacl)
        self.assertIsInstance(result, FeatureResult)
        self.assertTrue(result.names)
        self.assertTrue(all(name.startswith("geom.") for name in result.names))
        self.assertAlmostEqual(result.values["geom.a"], 5.64, places=6)
        self.assertEqual(result.to_frame().shape, (1, len(result.names)))
        self.assertEqual(result.to_vector().shape, (len(result.names),))

    def test_crystalnn_counts_six_neighbours_for_rocksalt(self):
        result = extract_features(self.nacl, sets=["cnn"])
        self.assertAlmostEqual(result.values["cnn.cn_mean"], 6.0)
        self.assertAlmostEqual(result.values["cnn.cn_min"], 6.0)
        self.assertAlmostEqual(result.values["cnn.cn_max"], 6.0)
        self.assertAlmostEqual(result.values["cnn.frac_cn_6"], 1.0)
        self.assertAlmostEqual(result.values["cnn.n_sites"], float(len(self.nacl)))

    def test_site_level_rows(self):
        result = FeatureExtractor(sets=["geometry", "cnn"], level="both")(self.licoo2)
        self.assertEqual(len(result.sites), len(self.licoo2))
        self.assertIn("site_index", result.site_names)
        self.assertIn("element", result.site_names)
        cobalt = [row for row in result.sites if row.get("element") == "Co"]
        self.assertEqual(len(cobalt), 3)
        self.assertAlmostEqual(cobalt[0]["cnn.cn"], 6.0)
        self.assertIn("geom.BAV", cobalt[0])

    def test_structure_only_level_has_no_sites(self):
        result = FeatureExtractor(sets=["geometry", "cnn"])(self.licoo2)
        self.assertEqual(result.sites, [])

    def test_include_and_exclude(self):
        kept = extract_features(
            self.nacl, sets=["geometry", "cnn"], include=["geom.a", "cnn.cn_*"]
        )
        self.assertIn("geom.a", kept.names)
        self.assertIn("cnn.cn_mean", kept.names)
        self.assertNotIn("geom.b", kept.names)

        dropped = extract_features(
            self.nacl, sets=["geometry", "cnn"], exclude=["cnn.*", "geom.a"]
        )
        self.assertFalse(any(name.startswith("cnn.") for name in dropped.names))
        self.assertNotIn("geom.a", dropped.names)

    def test_non_sodium_structure_does_not_raise(self):
        result = extract_features(self.licoo2)
        self.assertTrue(np.isnan(result.values["geom.Na_ratio"]))

    def test_metadata_records_structure_and_sets(self):
        result = extract_features(self.nacl, sets=["cnn"])
        self.assertEqual(result.metadata["sets"], ["cnn"])
        self.assertEqual(result.metadata["structure"]["formula"], "NaCl")
        self.assertEqual(result.metadata["structure"]["n_sites"], len(self.nacl))
        self.assertIn("cnn", result.metadata["providers"])

    def test_load_structure_accepts_poscar(self):
        path = Path(__file__).with_name("_tmp_test_POSCAR.vasp")
        try:
            poscar = Poscar(self.nacl)
            text = poscar.get_str() if hasattr(poscar, "get_str") else poscar.get_string()
            path.write_text(text, encoding="utf-8")
            loaded = load_structure(path)
            self.assertEqual(loaded.composition.reduced_formula, "NaCl")
            self.assertEqual(len(loaded), len(self.nacl))
            result = extract_features(str(path), sets=["cnn"])
            self.assertAlmostEqual(result.values["cnn.cn_mean"], 6.0)
        finally:
            if path.exists():
                path.unlink()

    def test_batch_isolates_a_bad_entry(self):
        frame = extract_many([self.nacl, "missing_file.cif"], sets=["cnn"])
        self.assertEqual(frame["status"].tolist(), ["ok", "error"])
        self.assertIn("FileNotFoundError", frame.loc[1, "error"])
        self.assertAlmostEqual(frame.loc[0, "cnn.cn_mean"], 6.0)

    def test_batch_returns_site_frame(self):
        frame, sites = extract_many(
            [self.nacl], sets=["geometry", "cnn"], level="both", return_sites=True
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(len(sites), len(self.nacl))
        self.assertIn("structure_index", sites.columns)

    def test_mlip_requires_a_model_path(self):
        with self.assertRaises(ValueError):
            FeatureExtractor(sets=["mlip"])

    def test_unavailable_optional_set_raises_import_error(self):
        if find_spec("matminer") is not None:
            self.skipTest("matminer is installed in this interpreter")
        with self.assertRaises(ImportError):
            FeatureExtractor(sets=["cnn_fp"])

    def test_available_sets_reports_prefix_and_availability(self):
        info = available_sets()
        self.assertEqual(info["geometry"]["prefix"], "geom.")
        self.assertIn("structure", info["geometry"]["levels"])
        self.assertEqual(info["cnn_fp"]["available"], find_spec("matminer") is not None)
        self.assertEqual(info["cnn_fp"]["prefix"], "cnn_fp.")


@unittest.skipUnless(
    find_spec("mace") is not None and find_spec("torch") is not None,
    "MACE is only installed in the py-mace interpreter",
)
class MaceDescriptorTests(unittest.TestCase):
    """Per-layer and per-element MACE descriptor selection (real checkpoint)."""

    @classmethod
    def setUpClass(cls):
        import os

        model = os.environ.get("MACE_MODEL")
        if not model:
            model = str(
                Path(__file__).resolve().parents[1]
                / "MLIP_model"
                / "mace"
                / "mace-omat-0-medium.model"
            )
        if not Path(model).is_file():
            raise unittest.SkipTest(f"MACE checkpoint not found: {model}")
        cls.model = model
        cls.nacl = rocksalt_nacl()

    def _extract(self, **kwargs):
        return extract_features(
            self.nacl, sets=["mlip"], encoder="mace", model_path=self.model, **kwargs
        )

    def test_first_layer_only(self):
        result = self._extract(mlip_layers=1)
        self.assertEqual(len(result.names), 128)
        self.assertTrue(all(name.startswith("mlip.d") for name in result.names))
        self.assertEqual(result.metadata["providers"]["mlip"]["layers"], [1])

    def test_second_layer_only_differs_from_first(self):
        first = self._extract(mlip_layers=1)
        second = self._extract(mlip_layers=2)
        self.assertEqual(len(second.names), 128)
        self.assertNotEqual(
            first.values["mlip.d000"], second.values["mlip.d000"]
        )

    def test_all_layers_concatenate(self):
        result = self._extract(mlip_layers="all")
        self.assertEqual(len(result.names), 256)
        self.assertEqual(result.metadata["providers"]["mlip"]["layers"], [1, 2])

    def test_element_mean_emits_one_block_per_element(self):
        result = self._extract(mlip_aggregate="element_mean")
        self.assertEqual(len(result.names), 512)
        self.assertIn("mlip.Na.d000", result.values)
        self.assertIn("mlip.Cl.d000", result.values)
        self.assertEqual(result.metadata["providers"]["mlip"]["elements"], ["Cl", "Na"])
        self.assertEqual(result.metadata["providers"]["mlip"]["aggregate"], "element_mean")


@unittest.skipUnless(
    find_spec("m3gnet") is not None
    and find_spec("maml") is not None
    and find_spec("tensorflow") is not None,
    "the M3GNet encoder needs the m3gnet/maml/TensorFlow interpreter",
)
class M3GNetEncoderTests(unittest.TestCase):
    """The M3GNet checkpoint path, loaded once per class."""

    @classmethod
    def setUpClass(cls):
        import os

        model = os.environ.get("M3GNET_MODEL") or str(
            Path(__file__).resolve().parents[1] / "MLIP_model" / "M3gnet"
        )
        if not Path(model).is_dir():
            raise unittest.SkipTest(f"M3GNet checkpoint not found: {model}")
        cls.model = model
        cls.extractor = FeatureExtractor(
            sets=["mlip"], encoder="m3gnet", model_path=model
        )
        cls.nacl = rocksalt_nacl()

    def test_structure_vector_has_64_named_columns(self):
        result = self.extractor(self.nacl)
        self.assertEqual(len(result.names), 64)
        self.assertEqual(result.names[0], "mlip.d000")
        self.assertEqual(result.metadata["providers"]["mlip"]["type"], "m3gnet_structure")
        self.assertEqual(result.metadata["providers"]["mlip"]["dimensions"], 64)
        self.assertTrue(all(value == value for value in result.values.values()))

    def test_mace_only_options_are_rejected(self):
        with self.assertRaises(ValueError):
            FeatureExtractor(
                sets=["mlip"], encoder="m3gnet", model_path=self.model, mlip_layers=1
            )
        with self.assertRaises(ValueError):
            FeatureExtractor(
                sets=["mlip"],
                encoder="m3gnet",
                model_path=self.model,
                mlip_aggregate="element_mean",
            )


if __name__ == "__main__":
    unittest.main()
