from __future__ import annotations

import tempfile
import unittest
import ast
from pathlib import Path

from nanolm.config import MODEL_PRESETS, ModelConfig, TrainConfig
from nanolm.corpus import CorpusLibrary, clean_text
from nanolm.model_planner import (assess_presets, exact_parameter_count,
                                  recommended_assessment,
                                  recommended_training_settings)


class ConfigurationTests(unittest.TestCase):
    def test_model_dimension_must_divide_heads(self):
        with self.assertRaises(ValueError):
            ModelConfig(vocab_size=300, dim=63, n_heads=4)

    def test_training_ranges_are_validated(self):
        with self.assertRaises(ValueError):
            TrainConfig(total_steps=0)
        with self.assertRaises(ValueError):
            TrainConfig(lr_max=1e-4, lr_min=2e-4)

    def test_extended_presets_are_available(self):
        self.assertIn("Large  (12 x 768, ctx 512)", MODEL_PRESETS)
        self.assertIn("XL     (16 x 1024, ctx 512)", MODEL_PRESETS)
        self.assertIn("XXL    (24 x 1280, ctx 512)", MODEL_PRESETS)

    def test_windows_install_pins_cuda_torch(self):
        requirements = (Path(__file__).parents[1] / "requirements.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch==2.12.1+cu126", requirements)
        self.assertIn("https://download.pytorch.org/whl/cu126", requirements)


class ModelPlannerTests(unittest.TestCase):
    def test_parameter_count_matches_architecture(self):
        counts = {
            name.split("(", 1)[0].strip(): exact_parameter_count(4096, preset)
            for name, preset in MODEL_PRESETS.items()
        }
        self.assertEqual(counts["Tiny"], 4_236_800)
        self.assertEqual(counts["Small"], 21_131_264)
        self.assertEqual(counts["Medium"], 42_154_240)
        self.assertGreater(counts["XXL"], 470_000_000)

    def test_recommendation_balances_corpus_and_vram(self):
        rows = assess_presets(4096, 5_000_000_000, 24.0, 16)
        recommended = recommended_assessment(rows)
        self.assertTrue(recommended.name.startswith("XL"))
        self.assertGreaterEqual(recommended.safe_batch, 1)

    def test_low_vram_rejects_xxl(self):
        rows = assess_presets(4096, 10_000_000_000, 4.0, 1)
        xxl = next(row for row in rows if row.name.startswith("XXL"))
        self.assertEqual(xxl.safe_batch, 0)
        self.assertEqual(xxl.vram_state, "poor")

    def test_applied_plan_has_complete_training_settings(self):
        row = recommended_assessment(assess_presets(4096, 1_000_000_000, 24.0, 16))
        settings = recommended_training_settings(row, 1_000_000_000)
        self.assertEqual(settings["preset"], row.name)
        self.assertGreaterEqual(settings["batch_size"], 1)
        self.assertGreaterEqual(settings["grad_accum"], 1)
        self.assertGreaterEqual(settings["total_steps"], 100)


class CorpusTests(unittest.TestCase):
    def test_custom_database_is_fully_isolated(self):
        with tempfile.TemporaryDirectory(prefix="nanolm_v4_test_") as name:
            root = Path(name)
            source = root / "source.txt"
            source.write_text("A useful document with enough unique content.", encoding="utf-8")
            library = CorpusLibrary(root / "store" / "corpus.db")
            row, _ = library.add_document(source)
            self.assertIsNotNone(row)
            self.assertTrue((root / "store" / "raw" / "00001.txt").exists())
            self.assertTrue((root / "store" / "documents" / "00001.txt").exists())

    def test_corpus_fingerprint_changes_with_active_set(self):
        with tempfile.TemporaryDirectory(prefix="nanolm_v4_test_") as name:
            root = Path(name)
            library = CorpusLibrary(root / "corpus.db")
            for index in range(2):
                source = root / f"source_{index}.txt"
                source.write_text(f"Document {index} has distinct content.", encoding="utf-8")
                library.add_document(source)
            before = library.corpus_fingerprint()
            library.set_active(2, False)
            self.assertNotEqual(before, library.corpus_fingerprint())

    def test_cleaning_preserves_paragraphs(self):
        value = clean_text("one\nline\n\nsecond\nparagraph")
        self.assertIn("\n\n", value)


class UISourceRegressionTests(unittest.TestCase):
    def test_library_tree_scrollbars_are_managed(self):
        """Protect the exact regression that made the supplied bar invisible."""
        ui_path = Path(__file__).parents[1] / "nanolm" / "ui.py"
        source = ui_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        app = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "App")
        method = next(node for node in app.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_build_library_tab")
        segment = ast.get_source_segment(source, method) or ""
        self.assertIn("self.tree_vscroll.grid", segment)
        self.assertIn("yscrollcommand=self.tree_vscroll.set", segment)
        self.assertIn("self.tree_hscroll.grid", segment)

    def test_overfit_gap_is_explicit_in_trainer_and_ui(self):
        root = Path(__file__).parents[1]
        trainer = (root / "nanolm" / "training.py").read_text(encoding="utf-8")
        ui = (root / "nanolm" / "ui.py").read_text(encoding="utf-8")
        self.assertIn('"train_loss": train_loss', trainer)
        self.assertIn('"gap": gap', trainer)
        self.assertIn('f"[eval] step={ev[\'step\']} train=', ui)
        self.assertIn("self.ax.fill_between", ui)

    def test_training_preflights_cuda_instead_of_silent_fallback(self):
        root = Path(__file__).parents[1]
        trainer = (root / "nanolm" / "training.py").read_text(encoding="utf-8")
        ui = (root / "nanolm" / "ui.py").read_text(encoding="utf-8")
        self.assertIn("def training_device()", trainer)
        self.assertIn("if _nvidia_gpu_detected():", trainer)
        self.assertIn("training_device()", ui)
        self.assertIn('messagebox.showerror("CUDA unavailable"', ui)


if __name__ == "__main__":
    unittest.main()
