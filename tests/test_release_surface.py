import re
import unittest
from pathlib import Path

import bio_sfm_trust

ROOT = Path(__file__).resolve().parents[1]


class ReleaseSurfaceTests(unittest.TestCase):
    def test_runtime_and_package_versions_match(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        project_section = pyproject.split("[project]", 1)[1].split("[", 1)[0]
        match = re.search(r'^version = "([^"]+)"$', project_section, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), bio_sfm_trust.__version__)

    def test_citation_version_matches_runtime(self):
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn(f"version: {bio_sfm_trust.__version__}", citation)

    def test_public_governance_files_exist(self):
        for filename in (
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "README.md",
            "SECURITY.md",
        ):
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / filename).is_file())

    def test_public_metadata_links_to_this_repository(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        repository = "https://github.com/jang1563/bio-sfm-trust-core"
        self.assertIn(f'Homepage = "{repository}"', pyproject)
        self.assertIn(f'Repository = "{repository}"', pyproject)
        self.assertIn(f'Issues = "{repository}/issues"', pyproject)

    def test_public_surface_has_no_workspace_or_internal_milestone_paths(self):
        forbidden = ("/Users/", "/home/", "Dropbox/", "designer milestone")
        paths = [ROOT / "README.md", ROOT / "pyproject.toml"]
        paths.extend((ROOT / "src" / "bio_sfm_trust").glob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for fragment in forbidden:
                with self.subTest(path=path.name, fragment=fragment):
                    self.assertNotIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
