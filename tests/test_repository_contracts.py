"""End-to-end contracts for committed evidence and publication artifacts."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import validate_repository as validator
from scripts.validate_repository import ValidationError, validate_repository


class RepositoryContractTests(unittest.TestCase):
    def test_plausible_but_tampered_effect_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            shutil.copytree(validator.DATA, data)
            effects = pd.read_csv(data / "joint_factorial_effects.csv")
            effects.loc[0, "F"] += 0.1
            effects.to_csv(data / "joint_factorial_effects.csv", index=False)
            with patch.object(validator, "DATA", data):
                with self.assertRaisesRegex(ValidationError, "joint_factorial_effects.csv"):
                    validator.validate_simulation_artifacts()

    def test_committed_repository_contracts(self) -> None:
        checks = validate_repository()
        self.assertEqual(
            [name for name, _detail in checks], ["simulation", "wisig", "publication", "ci"]
        )


if __name__ == "__main__":
    unittest.main()
