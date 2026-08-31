from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExecutiveCommerceBIContractTests(unittest.TestCase):
    def test_required_assets_exist(self) -> None:
        required = [
            "README.md",
            "KPI_DICTIONARY.md",
            "DASHBOARD_STORY.md",
            "dashboard_preview.svg",
            "refresh_verified_data.py",
            "power_bi/ExecutiveCommerce.pbip",
            "power_bi/ExecutiveCommerce.SemanticModel/definition.pbism",
            "power_bi/ExecutiveCommerce.SemanticModel/definition/model.tmdl",
            "power_bi/ExecutiveCommerce.Report/definition.pbir",
            "tableau/ExecutiveCommerce.twb",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_power_bi_bindings_and_two_page_story(self) -> None:
        pbip = json.loads((ROOT / "power_bi/ExecutiveCommerce.pbip").read_text())
        self.assertEqual(pbip["artifacts"][0]["report"]["path"], "ExecutiveCommerce.Report")
        pbir = json.loads((ROOT / "power_bi/ExecutiveCommerce.Report/definition.pbir").read_text())
        self.assertEqual(pbir["datasetReference"]["byPath"]["path"], "../ExecutiveCommerce.SemanticModel")
        pages = json.loads((ROOT / "power_bi/ExecutiveCommerce.Report/definition/pages/pages.json").read_text())
        self.assertIn(pages["activePageName"], pages["pageOrder"])
        self.assertEqual(2, len(pages["pageOrder"]))
        page_names = set()
        for page_id in pages["pageOrder"]:
            page = json.loads(
                (ROOT / f"power_bi/ExecutiveCommerce.Report/definition/pages/{page_id}/page.json").read_text()
            )
            page_names.add(page["displayName"])
        self.assertEqual({"Executive Overview", "Market & Operations"}, page_names)

    def test_power_bi_semantic_model_has_business_depth(self) -> None:
        model = (ROOT / "power_bi/ExecutiveCommerce.SemanticModel/definition/model.tmdl").read_text()
        for table in [
            "ExecutiveKPIs",
            "MonthlyPerformance",
            "CategoryPerformance",
            "DeliveryQuality",
            "StatePerformance",
            "PaymentBehaviour",
            "SellerOperations",
        ]:
            self.assertIn(f"ref table {table}", model)

        delivery = (ROOT / "power_bi/ExecutiveCommerce.SemanticModel/definition/tables/DeliveryQuality.tmdl").read_text()
        self.assertIn("measure 'Late Delivery Rate'", delivery)
        self.assertIn("measure 'Delivery Review Gap'", delivery)
        sellers = (ROOT / "power_bi/ExecutiveCommerce.SemanticModel/definition/tables/SellerOperations.tmdl").read_text()
        self.assertIn("measure 'Priority Sellers'", sellers)

    def test_all_pbir_json_is_valid(self) -> None:
        report = ROOT / "power_bi/ExecutiveCommerce.Report"
        for path in report.rglob("*.json"):
            with self.subTest(path=path):
                json.loads(path.read_text(encoding="utf-8"))

    def test_tableau_xml_and_dashboards(self) -> None:
        tree = ET.parse(ROOT / "tableau/ExecutiveCommerce.twb")
        dashboard_names = {item.attrib.get("name") for item in tree.findall("./dashboards/dashboard")}
        self.assertEqual({"Executive Commerce Dashboard", "Marketplace Explorer"}, dashboard_names)
        worksheet_names = {item.attrib.get("name") for item in tree.findall("./worksheets/worksheet")}
        self.assertEqual(
            {
                "Executive Pulse",
                "Monthly Trend",
                "Category Leaders",
                "Delivery Experience",
                "Regional Performance",
                "Payment Mix",
                "Seller Risk",
            },
            worksheet_names,
        )
        for worksheet in tree.findall("./worksheets/worksheet"):
            self.assertIsNotNone(worksheet.find("./table/view"))
            self.assertIsNotNone(worksheet.find("./table/panes/pane/mark"))
            self.assertIsNotNone(worksheet.find("./table/rows"))
            self.assertIsNotNone(worksheet.find("./table/cols"))

    def test_preview_uses_retained_values(self) -> None:
        preview = (ROOT / "dashboard_preview.svg").read_text()
        for value in ["98,199", "94,983", "R$13.49M", "3.03%", "13.24%", "4.28", "2.55"]:
            self.assertIn(value, preview)

    def test_generated_data_when_present(self) -> None:
        manifest_path = ROOT / "data/manifest.json"
        if not manifest_path.exists():
            self.skipTest("Generated dashboard extracts are created by the BI refresh workflow")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["verification"]["verification_pass"])
        self.assertEqual(98199, manifest["verification"]["headline"]["commercial_orders"])
        for filename in [
            "executive_kpis.csv",
            "monthly_performance.csv",
            "category_performance.csv",
            "delivery_review_summary.csv",
            "payment_behaviour.csv",
            "state_performance.csv",
            "seller_operational_review.csv",
            "tableau_dashboard_long.csv",
        ]:
            self.assertIn(filename, manifest["files"])


if __name__ == "__main__":
    unittest.main()
