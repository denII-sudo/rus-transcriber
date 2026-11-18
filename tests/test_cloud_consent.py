"""Облачная модель для конспекта — только по явному согласию человека (решение владельца 18.11.2026)."""
import ast
import re
import unittest
from pathlib import Path
from unittest import mock

import product_policy
import rt

ROOT = Path(__file__).resolve().parent.parent


class CloudConsentTests(unittest.TestCase):
    def test_policy_default_forbids_cloud(self):
        self.assertFalse(product_policy.DEFAULTS["allow_cloud_model"])
        self.assertFalse(product_policy.resolve({"require_conspect": True})["allow_cloud_model"])

    def test_mcp_start_workflow_default_forbids_cloud(self):
        tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_workflow")
        names = [a.arg for a in fn.args.args]
        defaults = dict(zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults))
        self.assertIs(defaults["allow_cloud_model"].value, False)

    def run_cli(self, *flags):
        with mock.patch.object(rt.core, "start_workflow", return_value={}) as start, \
             mock.patch.object(rt, "out"):
            rt.run(rt.build().parse_args(["workflow-start", "a.wav", "--conspect", *flags]))
        return start.call_args.args[1]["allow_cloud_model"]

    def test_cli_requires_explicit_cloud_flag(self):
        self.assertFalse(self.run_cli())
        self.assertTrue(self.run_cli("--cloud-model"))
        self.assertFalse(self.run_cli("--no-cloud-model"))
        self.assertFalse(self.run_cli("--cloud-model", "--no-cloud-model"))

    def test_screen_conspect_checkbox_is_the_consent(self):
        html = (ROOT / "workflow.html").read_text(encoding="utf-8")
        self.assertNotIn('id="cloud"', html)
        self.assertIn('allow_cloud_model: $("conspect").checked', html)
        label = re.search(r'id="conspect"[^>]*>([^<]*)<', html).group(1)
        self.assertIn("облак", label)
        self.assertNotRegex(html, r'id="conspect"[^>]*checked')


if __name__ == "__main__":
    unittest.main()
