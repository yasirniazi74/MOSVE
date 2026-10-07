import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_gene_label_release() -> list[dict[str, str]]:
    rows = []
    for filename, label in (("positive_genes.csv", "1"), ("negative_genes.csv", "0")):
        for row in read_csv(f"02_data/data_files/{filename}"):
            rows.append({"Gene": row["Gene"], "Label": label, **row})
    return rows


class ReleaseMetadataTests(unittest.TestCase):
    def test_gene_labels_have_expected_unique_counts_and_provenance(self) -> None:
        labels = read_gene_label_release()
        self.assertEqual(len(labels), 4420)
        self.assertEqual(len({row["Gene"] for row in labels}), 4420)
        self.assertEqual(sum(row["Label"] == "1" for row in labels), 1689)
        self.assertEqual(sum(row["Label"] == "0" for row in labels), 2731)
        for row in labels:
            self.assertIn(row["Label"], {"0", "1"})
            if row["Label"] == "1":
                self.assertTrue(row["Approved_Source_Membership"])

    def test_split_manifest_matches_gene_labels(self) -> None:
        labels = {row["Gene"]: row["Label"] for row in read_gene_label_release()}
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

        cohort = pd.DataFrame(read_gene_label_release())
        cohort["Label"] = cohort["Label"].astype(int)
        release_dir = ROOT / "02_data/data_files"
        module.validate_release_matches(cohort, release_dir)

        cohort.loc[0, "Label"] = 1 - cohort.loc[0, "Label"]
        with self.assertRaisesRegex(ValueError, "label_mismatches=1"):
            module.validate_release_matches(cohort, release_dir)

    def test_prediction_overlap_reports_shared_high_score_genes(self) -> None:
        path = ROOT / "01_code/gene_prediction_overlap.py"
        spec = importlib.util.spec_from_file_location("gene_prediction_overlap", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            inputs = {}
            values = {
                "TCGA": [("A", 0.9), ("B", 0.2)],
                "CCRCC": [("A", 0.8), ("C", 0.7)],
                "HNSCC": [("A", 0.6), ("C", 0.3)],
                "PCAWG": [("B", 0.8), ("C", 0.9)],
            }
            for dataset, rows in values.items():
                input_path = temp / f"{dataset.lower()}.csv"
                with input_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["Gene", "Score"])
                    writer.writerows(rows)
                inputs[dataset] = input_path

            membership_path, summary_path = module.write_overlap(
                inputs, temp / "output", 0.5, "Gene", "Score"
            )

            with membership_path.open(encoding="utf-8", newline="") as handle:
                membership = {row["Gene"]: row for row in csv.DictReader(handle)}
            self.assertEqual(membership["A"]["Dataset_Count"], "3")
            self.assertEqual(membership["A"]["Shared_At_Least_3"], "1")
            self.assertEqual(membership["C"]["Dataset_Count"], "2")
            with summary_path.open(encoding="utf-8", newline="") as handle:
                comparisons = {row["Comparison"]: row for row in csv.DictReader(handle)}
            self.assertEqual(comparisons["TCGA_vs_CCRCC"]["Overlap_Genes"], "1")

    def test_gene_file_exports_have_documented_counts_and_provenance(self) -> None:
        data_dir = ROOT / "02_data/data_files"
        files = read_csv("02_data/data_files/gene_file_manifest.csv")
        manifest = {row["File"]: row for row in files}
        self.assertEqual(int(manifest["verified_gene_labels.csv"]["Rows"]), 4420)
        self.assertEqual(int(manifest["predicted_driver_genes_score_ge_0.725.csv"]["Rows"]), 1165)
        self.assertEqual(int(manifest["candidate_genes_precision90.csv"]["Rows"]), 116)
        self.assertEqual(int(manifest["final_strict_novel_candidates.csv"]["Rows"]), 5)
        self.assertIn("5,418-gene", (ROOT / "README.md").read_text(encoding="utf-8"))

        with (data_dir / "predicted_driver_genes_score_ge_0.725.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            predictions = list(csv.DictReader(handle))
        self.assertTrue(all(float(row["MOSVE_Score"]) >= 0.725 for row in predictions))

        with (data_dir / "resource_overlap_genes.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            overlaps = list(csv.DictReader(handle))
        self.assertTrue(overlaps)
        self.assertTrue(all(row["Source_Flag"] for row in overlaps))


if __name__ == "__main__":
    unittest.main()
