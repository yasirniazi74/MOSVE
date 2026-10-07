from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path


DATASETS = ("TCGA", "CCRCC", "HNSCC", "PCAWG")


def read_scores(path: Path, gene_column: str, score_column: str) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or gene_column not in reader.fieldnames or score_column not in reader.fieldnames:
            raise ValueError(f"{path} must contain columns {gene_column!r} and {score_column!r}.")
        scores: dict[str, float] = {}
        for line_number, row in enumerate(reader, start=2):
            gene = (row.get(gene_column) or "").strip().upper()
            if not gene:
                raise ValueError(f"Missing gene symbol in {path} at line {line_number}.")
            try:
                score = float(row[score_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid score for {gene} in {path} at line {line_number}.") from error
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for {gene} in {path} at line {line_number}.")
            if gene in scores:
                raise ValueError(f"Duplicate gene symbol {gene!r} in {path}.")
            scores[gene] = score
    return scores


def write_overlap(inputs: dict[str, Path], output_dir: Path, threshold: float,
                  gene_column: str, score_column: str) -> tuple[Path, Path]:
    scores = {
        name: read_scores(path, gene_column, score_column)
        for name, path in inputs.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "prediction_overlap_membership.csv"
    summary_path = output_dir / "prediction_overlap_summary.csv"
    genes = sorted(set().union(*(set(values) for values in scores.values())))

    with membership_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["Gene"]
        for name in DATASETS:
            fields.extend([f"{name}_Score", f"{name}_Predicted"])
        fields.extend(["Dataset_Count", "Shared_At_Least_2", "Shared_At_Least_3"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gene in genes:
            row: dict[str, object] = {"Gene": gene}
            count = 0
            for name in DATASETS:
                score = scores[name].get(gene)
                predicted = score is not None and score >= threshold
                row[f"{name}_Score"] = "" if score is None else score
                row[f"{name}_Predicted"] = int(predicted)
                count += int(predicted)
            row["Dataset_Count"] = count
            row["Shared_At_Least_2"] = int(count >= 2)
            row["Shared_At_Least_3"] = int(count >= 3)
            writer.writerow(row)

    rows: list[dict[str, object]] = []
    predicted_sets = {
        name: {gene for gene, score in scores[name].items() if score >= threshold}
        for name in DATASETS
    }
    for name in DATASETS:
        rows.append({
            "Comparison": f"{name}_predicted",
            "Dataset_1": name,
            "Dataset_2": "",
            "Threshold": threshold,
            "Genes_Dataset_1": len(predicted_sets[name]),
            "Genes_Dataset_2": "",
            "Overlap_Genes": len(predicted_sets[name]),
            "Jaccard": "",
        })
    for first, second in itertools.combinations(DATASETS, 2):
        shared = predicted_sets[first] & predicted_sets[second]
        union = predicted_sets[first] | predicted_sets[second]
        rows.append({
            "Comparison": f"{first}_vs_{second}",
            "Dataset_1": first,
            "Dataset_2": second,
            "Threshold": threshold,
            "Genes_Dataset_1": len(predicted_sets[first]),
            "Genes_Dataset_2": len(predicted_sets[second]),
            "Overlap_Genes": len(shared),
            "Jaccard": len(shared) / len(union) if union else 0.0,
        })
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return membership_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare high-score MOSVE gene predictions across TCGA, CCRCC, HNSCC, and PCAWG."
    )
    for name in DATASETS:
        parser.add_argument(f"--{name.lower()}", required=True, type=Path, help=f"{name} CSV/TSV with gene and score columns")
    parser.add_argument("--gene-column", default="Gene")
    parser.add_argument("--score-column", default="Score")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("prediction_overlap"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    inputs = {name: getattr(args, name.lower()) for name in DATASETS}
    membership, summary = write_overlap(inputs, args.output_dir, args.threshold, args.gene_column, args.score_column)
    print(f"Membership table: {membership}")
    print(f"Pairwise summary: {summary}")


if __name__ == "__main__":
    main()
