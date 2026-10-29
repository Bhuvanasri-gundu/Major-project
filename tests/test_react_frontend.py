"""
tests/test_react_frontend.py — Tests verifying React 19 + TypeScript + Vite frontend configuration & assets.
"""

import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class TestReactFrontend(unittest.TestCase):
    def test_frontend_directory_exists(self):
        self.assertTrue(FRONTEND_DIR.exists(), "frontend directory does not exist")
        self.assertTrue((FRONTEND_DIR / "package.json").exists(), "package.json missing")
        self.assertTrue((FRONTEND_DIR / "vite.config.ts").exists(), "vite.config.ts missing")

    def test_react_19_dependency(self):
        pkg_path = FRONTEND_DIR / "package.json"
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg = json.load(f)
        deps = pkg.get("dependencies", {})
        self.assertIn("react", deps)
        self.assertIn("react-router-dom", deps)
        self.assertIn("@tanstack/react-query", deps)
        self.assertIn("lucide-react", deps)
        # Verify React 19
        react_version = deps["react"]
        self.assertTrue("19" in react_version, f"Expected React 19, found {react_version}")

    def test_compiled_dist_exists(self):
        dist_dir = FRONTEND_DIR / "dist"
        self.assertTrue(dist_dir.exists(), "dist/ directory does not exist. Run npm run build.")
        index_html = dist_dir / "index.html"
        self.assertTrue(index_html.exists(), "dist/index.html missing")
        with open(index_html, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn('id="root"', html)
        assets_dir = dist_dir / "assets"
        self.assertTrue(assets_dir.exists(), "dist/assets missing")
        js_files = list(assets_dir.glob("*.js"))
        css_files = list(assets_dir.glob("*.css"))
        self.assertGreater(len(js_files), 0, "Compiled JavaScript bundle missing")
        self.assertGreater(len(css_files), 0, "Compiled CSS stylesheet missing")

    def test_two_primary_screens_implemented(self):
        pages_dir = FRONTEND_DIR / "src" / "pages"
        self.assertTrue((pages_dir / "GeneratePage.tsx").exists(), "Screen 1: GeneratePage.tsx missing")
        self.assertTrue((pages_dir / "WorkspacePage.tsx").exists(), "Screen 2: WorkspacePage.tsx missing")


if __name__ == "__main__":
    unittest.main()
