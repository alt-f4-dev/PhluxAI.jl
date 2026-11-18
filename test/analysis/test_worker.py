import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from fixtures import make_fixtures

WORKER_PATH = pathlib.Path(__file__).parents[2] / "src" / "analysis" / "worker.py"
spec = importlib.util.spec_from_file_location("analysis_worker", WORKER_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
LIMITS = dict(max_pages=16, max_document_pages=10000, max_cells=10000, max_dpi=300, max_pixels=16000000,
              max_nodes=1000, max_attribute_bytes=4096, max_elements=100000,
              max_decoded_bytes=16 * 1024 * 1024, max_output_bytes=1000000)


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temporary.name)
        make_fixtures(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_analysis(self, operation, filename=None, limits=None, **arguments):
        filename = filename or ("sample.pdf" if operation.startswith("pdf_") else "sample.h5")
        request = dict(operation=operation, arguments=arguments, limits=LIMITS | (limits or {}))
        return worker.analyze(request, str(self.root / filename))

    def assert_failure(self, expected, operation, **arguments):
        with self.assertRaises(worker.AnalysisError) as caught:
            self.run_analysis(operation, **arguments)
        self.assertEqual(caught.exception.code, expected)

    def test_pdf_metadata_text_and_tables(self):
        metadata = self.run_analysis("pdf_metadata")
        self.assertEqual(metadata["data"]["page_count"], 3)
        self.assertEqual(metadata["data"]["metadata"]["Title"], "Analysis fixture")
        extracted = self.run_analysis("pdf_text", pages=[1, 3])
        self.assertIn("PhluxAI deterministic text 12345", extracted["data"][0]["text"])
        self.assertFalse(extracted["data"][1]["has_text"])
        self.assertEqual(extracted["evidence"][0]["locator"]["page"], 1)
        table = self.run_analysis("pdf_tables", pages=[2])
        self.assertEqual(table["data"][0]["tables"][0]["cells"], [["A", "B"], ["2", "4"]])
        self.assertEqual(table["evidence"][0]["scheme"], "pdf_region")

    def test_ocr_known_content(self):
        result = self.run_analysis("pdf_ocr", pages=[3], dpi=216)
        self.assertEqual(result["data"][0]["text"].strip(), "PHLUXAI SAMPLE 12345")
        self.assertTrue(all(word["confidence"] >= 0 for word in result["data"][0]["words"]))
        self.assertEqual(result["method"], "poppler+tesseract")
        self.assertIn("tesseract", result["versions"])

    def test_pdf_bounds_and_encryption(self):
        self.assert_failure("invalid_request", "pdf_text", pages=[0])
        self.assert_failure("invalid_request", "pdf_text", pages=[4])
        self.assert_failure("invalid_request", "pdf_text", pages=[1, 1])
        self.assert_failure("invalid_request", "pdf_text", pages=[{}])
        self.assert_failure("limit_exceeded", "pdf_text", limits={"max_pages": 1}, pages=[1, 2])
        self.assert_failure("limit_exceeded", "pdf_metadata", limits={"max_document_pages": 1})
        self.assert_failure("limit_exceeded", "pdf_tables", pages=[2], limits={"max_cells": 1})
        self.assert_failure("limit_exceeded", "pdf_ocr", pages=[3], limits={"max_pixels": 1})
        self.assert_failure("encrypted_source", "pdf_text", filename="encrypted.pdf")
        self.assertEqual(self.run_analysis("pdf_metadata", filename="empty.pdf")["data"]["page_count"], 0)

    def test_hyperslab_matches_independent_numpy(self):
        selection = [dict(start=2, stop=9, step=2), dict(start=1, stop=6, step=2)]
        result = self.run_analysis("hdf5_read", dataset="/data", selection=selection)
        expected = np.arange(60).reshape(10, 6)[2:9:2, 1:6:2].tolist()
        self.assertEqual(result["data"]["values"], expected)
        self.assertEqual(result["data"]["shape"], [4, 3])
        self.assertEqual(result["data"]["elements_read"], 12)
        stats = self.run_analysis("hdf5_statistics", dataset="/data", selection=selection)["data"]
        self.assertAlmostEqual(stats["mean"], np.mean(expected))
        self.assertEqual(stats["min"], np.min(expected))
        self.assertEqual(stats["max"], np.max(expected))
        self.assertEqual(stats["count"], 12)

    def test_hdf5_types_shapes_and_nonfinite(self):
        scalar = self.run_analysis("hdf5_read", dataset="/scalar")["data"]
        self.assertEqual(scalar["values"], 2**60 + 1)
        self.assertEqual(self.run_analysis("hdf5_read", dataset="/unsigned_scalar")["data"]["values"], 2**63 + 1)
        for path in ("/empty", "/null"):
            self.assertEqual(self.run_analysis("hdf5_statistics", dataset=path)["data"]["count"], 0)
        stats = self.run_analysis("hdf5_statistics", dataset="/nonfinite")["data"]
        self.assertEqual((stats["count"], stats["finite_count"], stats["nan_count"]), (5, 2, 1))
        self.assertEqual(stats["mean"], 2)
        self.assertEqual(stats["positive_infinity_count"], 1)
        self.assertEqual(stats["negative_infinity_count"], 1)
        self.assertEqual(self.run_analysis("hdf5_statistics", dataset="/largefloat")["data"]["mean"], 1e308)
        self.assertEqual(self.run_analysis("hdf5_read", dataset="/nonfinite")["data"]["values"], [1, 3, None, None, None])
        metadata = self.run_analysis("hdf5_metadata", dataset="/data")["data"]
        self.assertEqual(metadata["attributes"]["units"], "meV")
        self.assertIn("omitted", metadata["attributes"]["oversized"])
        self.assertIn("omitted", metadata["attributes"]["complex"])
        self.assertTrue(self.run_analysis("hdf5_metadata", dataset="/virtual")["data"]["virtual"])

    def test_hdf5_references_and_filters(self):
        for path in ("/soft", "/external_link", "/external_storage", "/virtual"):
            self.assert_failure("unauthorized_reference", "hdf5_read", dataset=path)
        for path in ("/strings", "/variable"):
            self.assert_failure("unsupported_type", "hdf5_read", dataset=path)
        self.assert_failure("unsupported_filter", "hdf5_read", dataset="/unknown_filter")
        hierarchy = self.run_analysis("hdf5_hierarchy")["data"]
        self.assertLess(len(hierarchy), 50)
        self.assertEqual(next(item for item in hierarchy if item["path"] == "/soft")["kind"], "blocked_link")

    def test_hdf5_bounds(self):
        self.assert_failure("invalid_request", "hdf5_read", dataset="/../data")
        self.assert_failure("invalid_request", "hdf5_read", dataset="/missing")
        self.assert_failure("invalid_request", "hdf5_read", dataset="/data", selection=[{}])
        self.assert_failure("invalid_request", "hdf5_read", dataset="/data", selection=[{"step": 0}, {}])
        self.assert_failure("invalid_request", "hdf5_read", dataset="/data", selection=[{"stop": 20}, {}])
        self.assert_failure("limit_exceeded", "hdf5_read", dataset="/data", limits={"max_elements": 2})
        self.assert_failure("limit_exceeded", "hdf5_hierarchy", limits={"max_nodes": 1})
        self.assert_failure("limit_exceeded", "hdf5_read", dataset="/data", selection=[{"stop": 1}, {"stop": 1}],
                            limits={"max_decoded_bytes": 8})

    def test_tiny_read_of_huge_dataset_uses_actual_slice(self):
        original = h5py.Dataset.__getitem__
        selections = []

        def guarded(dataset, key, *args, **kwargs):
            self.assertEqual(key, (slice(100, 110, 1),))
            selections.append(key)
            return original(dataset, key, *args, **kwargs)

        with patch.object(h5py.Dataset, "__getitem__", guarded):
            result = self.run_analysis("hdf5_read", dataset="/huge", selection=[dict(start=100, stop=110)])
        self.assertEqual(result["data"]["values"], [7] * 10)
        self.assertEqual(len(selections), 1)

    def test_fixed_width_types_and_three_dimensional_selections(self):
        selection = [dict(start=0, stop=2), dict(start=1, stop=3), dict(start=0, stop=4, step=2)]
        for dtype in ("i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8", "bool", ">i4"):
            with self.subTest(dtype=dtype):
                expected = np.arange(24).reshape(2, 3, 4).astype(dtype)[:, 1:3, 0:4:2]
                result = self.run_analysis("hdf5_read", dataset="/types/" + dtype, selection=selection)
                self.assertEqual(result["data"]["values"], expected.tolist())
                stats = self.run_analysis("hdf5_statistics", dataset="/types/" + dtype, selection=selection)["data"]
                self.assertAlmostEqual(stats["mean"], float(expected.mean()))

    def test_cli_failures_output_bound_and_recovery(self):
        for filename, operation in (("malformed.pdf", "pdf_text"), ("malformed.h5", "hdf5_read")):
            request = dict(operation=operation, arguments={}, limits=LIMITS)
            result = subprocess.run([sys.executable, "-I", str(WORKER_PATH), json.dumps(request), str(self.root / filename)],
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["error"], "malformed_source")
        request = dict(operation="pdf_text", arguments={}, limits=LIMITS | {"max_output_bytes": 1})
        result = subprocess.run([sys.executable, "-I", str(WORKER_PATH), json.dumps(request), str(self.root / "sample.pdf")],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["error"], "limit_exceeded")
        self.assertTrue(self.run_analysis("pdf_text")["ok"])


if __name__ == "__main__":
    unittest.main()
