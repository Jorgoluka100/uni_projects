import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extensions"))

from export_albania_all_age_outputs import export_outputs


class AlbaniaOutputExportTests(unittest.TestCase):
    def test_exports_reconciled_machine_readable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            written = export_outputs(
                Path(directory),
                regional_budget_eur=10_000_000,
                n_sims=25,
                seed=11,
            )
            self.assertEqual(set(written), {
                "prefecture_age_profile.csv",
                "allocation_scenario_comparison.csv",
                "allocation_weight_sensitivity.csv",
                "scenario_metadata.json",
            })
            profile = __import__("pandas").read_csv(written["prefecture_age_profile.csv"])
            comparison = __import__("pandas").read_csv(written["allocation_scenario_comparison.csv"])
            sensitivity = __import__("pandas").read_csv(written["allocation_weight_sensitivity.csv"])
            metadata = json.loads(Path(written["scenario_metadata.json"]).read_text(encoding="utf-8"))

            self.assertEqual(len(profile), 12)
            self.assertEqual(int(profile["population"].sum()), 2_402_113)
            self.assertEqual(len(comparison), 12)
            self.assertNotIn("older_age_focused_share", comparison.columns)
            self.assertAlmostEqual(comparison["all_age_allocation_eur"].sum(), 10_000_000)
            self.assertEqual(len(sensitivity), 12)
            self.assertEqual(metadata["scenario_scope"], "balanced_all_age")
            self.assertEqual(metadata["age_cohorts"], ["0-14", "15-64", "65+"])
            self.assertEqual(metadata["sensitivity_simulations"], 25)
            self.assertEqual(metadata["random_seed"], 11)
            self.assertIn("assumptions", metadata["interpretation"])

    def test_rejects_negative_budget_and_zero_simulations(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_outputs(Path(directory), regional_budget_eur=-1)
            with self.assertRaises(ValueError):
                export_outputs(Path(directory), n_sims=0)


if __name__ == "__main__":
    unittest.main()
