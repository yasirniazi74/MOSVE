import csv
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ReleaseMetadataTests(unittest.TestCase):
    def test_gene_labels_have_expected_unique_counts_and_provenance(self) -> None:
        labels = read_csv("02_data/labels/gene_labels.csv")
        self.assertEqual(len(labels), 4420)
        self.assertEqual(len({row["Gene"] for row in labels}), 4420)
        self.assertEqual(sum(row["Label"] == "1" for row in labels), 1689)
        self.assertEqual(sum(row["Label"] == "0" for row in labels), 2731)
        for row in labels:
            self.assertIn(row["Label"], {"0", "1"})
            if row["Label"] == "1":
                self.assertTrue(row["Approved_Source_Membership"])

    def test_split_manifest_matches_gene_labels(self) -> None:
        labels = {row["Gene"]: row["Label"] for row in read_csv("02_data/labels/gene_labels.csv")}
        splits = read_csv("02_data/manifests/split_membership.csv")
        self.assertEqual(len(splits), 4420)
        self.assertEqual({row["Gene"] for row in splits}, set(labels))
        self.assertTrue(all(row["Label"] == labels[row["Gene"]] for row in splits))
        self.assertEqual(
            {name: sum(row["Split"] == name for row in splits)
             for name in {"development", "reserved_holdout"}},
            {"development": 3536, "reserved_holdout": 884},
        )

    def test_label_policy_counts_match_released_data(self) -> None:
        policy = json.loads((ROOT / "02_data/metadata/label_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["counts"], {"genes": 4420, "positive": 1689, "negative": 2731})

    def test_model_source_rejects_a_cohort_different_from_release(self) -> None:
        path = ROOT / "01_code/model_and_candidates.py"
        spec = importlib.util.spec_from_file_location("mosve_training", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        cohort = pd.DataFrame(read_csv("02_data/labels/gene_labels.csv"))
        cohort["Label"] = cohort["Label"].astype(int)
        module.validate_release_matches(cohort, ROOT / "02_data/labels/gene_labels.csv")

        cohort.loc[0, "Label"] = 1 - cohort.loc[0, "Label"]
        with self.assertRaisesRegex(ValueError, "label_mismatches=1"):
            module.validate_release_matches(cohort, ROOT / "02_data/labels/gene_labels.csv")


if __name__ == "__main__":
    unittest.main()
