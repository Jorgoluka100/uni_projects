from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ExecutiveCommerceBIContractTests(unittest.TestCase):
    def test_required_assets_exist(self) -> None:
        required = [
            "README.md", "KPI_DICTIONARY.md", "DASHBOARD_STORY.md", "dashboard_preview.svg",
            "power_bi/ExecutiveCommerce.pbip",
            "power_bi/ExecutiveCommerce.SemanticModel/definition.pbism",
            "power_bi/ExecutiveCommerce.SemanticModel/definition/model.tmdl",
            "power_bi/ExecutiveCommerce.Report/definition.pbir",
            "tableau/ExecutiveCommerce.twb",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_power_bi_bindings(self) -> None:
        pbip = json.loads((ROOT / "power_bi/ExecutiveCommerce.pbip").read_text())
        self.assertEqual(pbip["artifacts"][0]["report"]["path"], "ExecutiveCommerce.Report")
        pbir = json.loads((ROOT / "power_bi/ExecutiveCommerce.Report/definition.pbir").read_text())
        self.assertEqual(pbir["datasetReference"]["byPath"]["path"], "../ExecutiveCommerce.SemanticModel")
        pages = json.loads((ROOT / "power_bi/ExecutiveCommerce.Report/definition/pages/pages.json").read_text())
        self.assertIn(pages["activePageName"], pages["pageOrder"])

    def test_tableau_xml_and_dashboard(self) -> None:
        tree = ET.parse(ROOT / "tableau/ExecutiveCommerce.twb")
        dashboard_names = [item.attrib.get("name") for item in tree.findall("./dashboards/dashboard")]
        self.assertIn("Executive Commerce Dashboard", dashboard_names)
        worksheet_names = {item.attrib.get("name") for item in tree.findall("./worksheets/worksheet")}
        self.assertEqual({"Executive Pulse", "Monthly Trend", "Category Leaders", "Delivery Experience"}, worksheet_names)

    def test_preview_uses_retained_values(self) -> None:
        preview = (ROOT / "dashboard_preview.svg").read_text()
        for value in ["98,199", "94,983", "R$13.49M", "3.03%", "13.24%", "4.28", "2.55"]:
            self.assertIn(value, preview)


if __name__ == "__main__":
    unittest.main()
