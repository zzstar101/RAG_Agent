import unittest

from agent.tools import middleware
from model import factory
from utils import logger_handler


class ReportContextStateMachineTests(unittest.TestCase):
    def test_report_context_can_activate_and_expire(self):
        context = {}

        middleware.activate_report_context(context, now_ts=100.0, ttl_seconds=10, trace_id="t1")
        self.assertTrue(middleware.is_report_context_active(context, now_ts=105.0))
        self.assertFalse(middleware.is_report_context_active(context, now_ts=111.0))
        self.assertEqual(context.get("report_last_exit_reason"), "ttl_expired")


class ModelFactoryHealthCheckTests(unittest.TestCase):
    def test_startup_check_contains_required_dimensions(self):
        result = factory.run_startup_checks(connectivity_host="localhost", connectivity_port=443, timeout_seconds=0.01)

        self.assertIn("model_name", result)
        self.assertIn("credential", result)
        self.assertIn("connectivity", result)


class LoggerTraceTests(unittest.TestCase):
    def test_trace_id_lifecycle(self):
        logger_handler.clear_trace_id()
        trace_id = logger_handler.ensure_trace_id()

        self.assertTrue(trace_id)
        self.assertEqual(trace_id, logger_handler.get_trace_id())

        logger_handler.clear_trace_id()
        self.assertEqual(logger_handler.get_trace_id(), "-")


if __name__ == "__main__":
    unittest.main()
