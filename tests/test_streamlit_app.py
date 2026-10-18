"""
test_streamlit_app.py — Tests for Streamlit app syntax, imports, and data loaders.
"""

import unittest
import py_compile
from pathlib import Path
import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestStreamlitApp(unittest.TestCase):
    def test_streamlit_app_compiles(self):
        app_path = PROJECT_ROOT / "streamlit_app.py"
        self.assertTrue(app_path.exists(), "streamlit_app.py does not exist")
        try:
            py_compile.compile(str(app_path), doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"streamlit_app.py has syntax errors: {e}")

    def test_streamlit_loads_phase1_data(self):
        from streamlit_app import load_phase1_data
        data = load_phase1_data()
        self.assertIsNotNone(data)
        self.assertIn("gaps", data)
        self.assertIsNotNone(data["gaps"])
        self.assertGreater(len(data["gaps"]), 0)

    def test_streamlit_loads_phase2_data(self):
        from streamlit_app import load_phase2_data
        data = load_phase2_data()
        self.assertIsNotNone(data)
        self.assertIn("hypotheses", data)
        self.assertIn("verification", data)
        self.assertIn("rankings", data)


if __name__ == "__main__":
    unittest.main()
