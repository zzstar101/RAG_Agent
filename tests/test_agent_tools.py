import json
import unittest
from unittest.mock import patch

from agent.tools import agent_tools


class FetchExternalDataTests(unittest.TestCase):
    def setUp(self):
        self._original_external_data = agent_tools.external_data
        agent_tools.external_data = {}

    def tearDown(self):
        agent_tools.external_data = self._original_external_data

    def test_fetch_external_data_returns_structured_json_when_record_exists(self):
        agent_tools.external_data = {
            "1001": {
                "2025-01": {
                    "特征": "高频清扫",
                    "效率": "92%",
                    "耗材": "正常",
                    "对比": "较上月提升"
                }
            }
        }

        with patch.object(agent_tools, "generate_external_data", return_value=None):
            result = agent_tools.fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})

        payload = json.loads(result)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["data"]["效率"], "92%")
        self.assertEqual(payload["user_id"], "1001")
        self.assertEqual(payload["month"], "2025-01")

    def test_fetch_external_data_marks_missing_record_as_no_data(self):
        with patch.object(agent_tools, "generate_external_data", return_value=None):
            result = agent_tools.fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})

        payload = json.loads(result)
        self.assertEqual(payload["status"], "no_data")
        self.assertIsNone(payload["data"])
        self.assertEqual(payload["user_id"], "1001")
        self.assertEqual(payload["month"], "2025-01")

    def test_fetch_external_data_marks_loader_failure_as_error(self):
        with patch.object(agent_tools, "generate_external_data", side_effect=FileNotFoundError("missing records file")):
            result = agent_tools.fetch_external_data.invoke({"user_id": "1001", "month": "2025-01"})

        payload = json.loads(result)
        self.assertEqual(payload["status"], "error")
        self.assertIsNone(payload["data"])
        self.assertEqual(payload["user_id"], "1001")
        self.assertEqual(payload["month"], "2025-01")
        self.assertIn("FileNotFoundError", payload["message"])


if __name__ == "__main__":
    unittest.main()
