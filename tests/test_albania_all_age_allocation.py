import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions"))
from albania_all_age_allocation import (
    AGE_COLUMNS,
    TOTAL_POPULATION,
    allocate_budget,
    build_age_profile,
    calculate_all_age_need_shares,
    compare_allocation_scenarios,
    run_weight_sensitivity,
)


class AlbaniaAllAgeAllocationTests(unittest.TestCase):
    def test_census_totals_reconcile(self):
        profile = build_age_profile()
        self.assertEqual(len(profile), 12)
        self.assertEqual(int(profile["population"].sum()), TOTAL_POPULATION)
        self.assertEqual(profile[AGE_COLUMNS].sum().to_dict(), {
            "age_0_14": 373929,
            "age_15_64": 1555080,
            "age_65_plus": 473104,
        })

    def test_age_groups_cover_each_prefecture(self):
        profile = build_age_profile()
        self.assertTrue(
            np.array_equal(
                profile["population"].to_numpy(),
                profile[AGE_COLUMNS].sum(axis=1).to_numpy(),
            )
        )
        self.assertTrue((profile["total_dependency_ratio"] >= 0).all())

    def test_all_age_shares_sum_to_one_and_respect_floor(self):
        profile = build_age_profile()
        shares = calculate_all_age_need_shares(profile, equity_floor=0.15)
        self.assertAlmostEqual(float(shares.sum()), 1.0)
        self.assertTrue((shares >= 0.15 / 12).all())

    def test_budget_is_conserved(self):
        profile = build_age_profile()
        shares = calculate_all_age_need_shares(profile)
        allocation = allocate_budget(25_000_000, shares)
        self.assertAlmostEqual(allocation["allocation_eur"].sum(), 25_000_000)

    def test_scenario_comparison_and_sensitivity_are_reproducible(self):
        comparison = compare_allocation_scenarios(10_000_000)
        self.assertEqual(len(comparison), 12)
        self.assertAlmostEqual(comparison["all_age_allocation_eur"].sum(), 10_000_000)
        first = run_weight_sensitivity(10_000_000, n_sims=30, seed=7)
        second = run_weight_sensitivity(10_000_000, n_sims=30, seed=7)
        self.assertTrue(first.equals(second))
        self.assertTrue((first["p10_eur"] <= first["median_eur"]).all())
        self.assertTrue((first["median_eur"] <= first["p90_eur"]).all())


if __name__ == "__main__":
    unittest.main()
