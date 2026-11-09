"""Offline regression tests for the analysis tools."""

import json
import unittest

import pandas as pd

from handson import build_tools


class AnalysisToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = build_tools(pd.DataFrame({
            "group": ["A", "A", "B", None],
            "value": [2.0, 4.0, None, 8.0],
        }))

    def test_inspection_counts_missing_values(self) -> None:
        result = json.loads(self.tools[0].invoke({}))
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["columns"], 2)
        self.assertEqual(result["missing"]["value"], 1)

    def test_summary_uses_nonmissing_observations(self) -> None:
        result = json.loads(self.tools[1].invoke({"column": "value"}))
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mean"], 14 / 3)
        self.assertEqual(result["50%"], 4)

    def test_groups_include_missing_keys_and_report_valid_counts(self) -> None:
        result = json.loads(self.tools[2].invoke({"group": "group", "value": "value"}))["data"]
        group_a = next(row for row in result if row["group"] == "A")
        self.assertEqual(group_a["count"], 2)
        self.assertEqual(group_a["mean"], 3)
        self.assertTrue(any(row["group"] is None for row in result))
        group_b = next(row for row in result if row["group"] == "B")
        self.assertEqual(group_b["count"], 0)
        self.assertIsNone(group_b["mean"])

    def test_invalid_columns_return_an_explicit_error(self) -> None:
        self.assertTrue(self.tools[1].invoke({"column": "missing"}).startswith("오류"))
        self.assertTrue(self.tools[2].invoke({"group": "group", "value": "group"}).startswith("오류"))

    def test_identifier_sized_groups_are_not_sent_to_model(self) -> None:
        tools = build_tools(pd.DataFrame({"id": range(51), "value": range(51)}))
        result = tools[2].invoke({"group": "id", "value": "value"})
        self.assertTrue(result.startswith("오류"))


if __name__ == "__main__":
    unittest.main()
