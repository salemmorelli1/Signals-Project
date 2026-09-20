"""End-to-end contracts for committed evidence and publication artifacts."""

from __future__ import annotations

import unittest

from scripts.validate_repository import validate_repository


class RepositoryContractTests(unittest.TestCase):
    def test_committed_repository_contracts(self) -> None:
        checks = validate_repository()
        self.assertEqual(
            [name for name, _detail in checks], ["simulation", "wisig", "publication", "ci"]
        )


if __name__ == "__main__":
    unittest.main()
