# MOSVE

This repository is a source-and-metadata release for the MOSVE paper. It is
structured around the analysis order:

1. **Feature extraction** for the TCGA development data.
2. **Strict-label cohort construction and MOSVE model evaluation.**
3. **Resource-filtered candidate-gene prioritization.**
4. **External validation** using PCAWG and CPTAC inputs.

The repository intentionally contains **no raw cohort files, downloaded
database files, or full extracted-feature matrices**. Those files are large,
may have separate access/licensing terms, and must be obtained independently.
The only bundled gene-level labels are the paper's 4,420-gene release, with
label provenance and the fixed development/holdout assignment.

## Contents

- `01_code/feature_extraction.py`: sequence/omics, STRING graph, and engineered
  feature stages.
- `01_code/model_and_candidates.py`: strict-label model training.
- `01_code/novel_candidate_score.py`: resource-filtered novel candidate scores.
- `01_code/gene_prediction_overlap.py`: shared predictions across TCGA, CCRCC,
  HNSCC, and PCAWG.
- `05_external_validation/pcawg_validation.py`: PCAWG feature-building and
  validation tasks.
- `05_external_validation/cptac_data.py`: CPTAC input download and feature
  preparation.
- `05_external_validation/cptac_validation.py`: CPTAC validation tasks.
- `02_data/data_files/positive_genes.csv` and `negative_genes.csv`: the 4,420
  released genes, split by class, with evidence-source indicators.
- `02_data/data_files/verified_gene_labels.csv` and
  `verified_gene_labels.txt`: all 4,420 final labels in CSV and tab-separated
  text form. `verified_positive_genes.txt` lists all positive symbols; the
  `verified_positive_genes_<SOURCE>.txt` files list positive genes supported
  by each approved source. Source lists can overlap.
- `02_data/data_files/all_gene_labels_combined.csv`: original 5,455-row
  gene-level source ledger. It is provenance, not the final curated label set.
- `02_data/data_files/predicted_driver_genes_score_ge_0.725.csv`: existing
  out-of-fold driver predictions at the documented 0.725 cutoff.
- `02_data/data_files/candidate_genes_precision90.csv` and
  `final_strict_novel_candidates.csv`: the prior precision-90 candidate set
  and the five candidates remaining after external novelty checks.
- `02_data/data_files/resource_overlap_genes.csv`: per-gene hits in the
  recorded driver-resource overlap checks. Each row preserves its source flag.
- `02_data/data_files/gene_file_manifest.csv`: counts, scope, and provenance
  for the released gene files.
- `02_data/manifests/split_membership.csv`: fixed gene-level train/holdout
  allocation (no feature values).
- `02_data/manifests/input_manifest.csv`: required external inputs, expected
  paths, and source portals.
- `02_data/metadata/label_policy.json`: label rules, counts, and split
  parameters.

Generated outputs and downloaded data are ignored by Git. Do not add feature
matrices, cohort-level raw files, or other derived feature tables to a public
commit.

The workflow scripts group related operations behind named commands. PCAWG
and CPTAC raw data are not bundled; dataset-specific sources and expected
locations are listed in `02_data/manifests/input_manifest.csv`. Raw-input and
generated-output folders are created when needed.

## Label release

The two class files contain 4,420 unique symbols, split by filename:

- `positive_genes.csv`: 1,689 positive genes
- `negative_genes.csv`: 2,731 reference-negative genes

The predictions and candidate files are exported from the existing research
results, rather than recalculated by the consolidated scripts. Their source
run used a 5,418-gene cohort and the predicted-driver file uses a 0.725
threshold; these historical scores are not a new evaluation of the released
4,420-gene cohort. See `gene_file_manifest.csv` for source files and row counts.
The original source ledger has 5,455 rows and includes source-level labels that
did not necessarily survive the final release policy. Use
`verified_gene_labels.csv` as the authoritative 4,420-gene label release.

Positive genes require direct support from at least one of CGC, OncoKB,
OnGene, or NCG. The strict cohort construction relabels 194 documented
negative conflicts with direct evidence, excludes 596 CancerMine-dependent
positives and 143 unsupported positives, and removes 259 negatives found in
the approved driver resources. The row-level audit and raw evidence tables
were not copied into this repository.

The split manifest preserves the experiment's seed-42 assignment:

- development: 3,536 genes
- reserved holdout: 884 genes

These labels and assignments are supplied as a compact reference artifact;
they do not replace the source ledger and databases required to reconstruct
the cohort from scratch.

## Setup

Use Python 3.10 or later. From this repository root, create an isolated
environment and install:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Feature extraction calls NCBI E-utilities. Set a valid contact email before
running the sequence-fetching stage:

```powershell
$env:NCBI_EMAIL = "you@example.org"
```

## Required inputs and execution

1. Review `02_data/manifests/input_manifest.csv`. Download the named files
   directly from their provider and place them under the specified repository
   paths. Some sources (including COSMIC/OncoKB and some CPTAC data) may require
   registration or acceptance of provider terms. The download manifest records
   provider portals, not redistribution permission.
2. Obtain the upstream `enhanced_sequence_dataset_multisource.csv` input from
   the project/source archive. It is not bundled; no public download location
   for this project-specific feature input was established. The original
   gene-level label source ledger is included in `02_data/data_files/`.
3. Run the feature extraction stages in order with the working directory set
   to `02_data/final_model_inputs`:

   ```powershell
   Set-Location .\02_data\final_model_inputs
   python ..\..\01_code\feature_extraction.py all
   Set-Location ..\..
   ```

   The first stage expects the upstream base matrix and raw assets listed in
   the manifest. It writes intermediate feature matrices locally; they are
   ignored by Git.
4. Run strict label construction and the primary model:

   ```powershell
   python .\01_code\model_and_candidates.py train
   ```

   It creates `04_strict_verified_4420_experiment/` locally. The script checks
   the expected 4,420/1,689/2,731 cohort counts and compares every gene and
   label against the combined positive and negative gene files in
   `02_data/data_files/`. It stops if the reconstructed cohort differs, then
   writes feature-bearing experiment artifacts locally; keep those generated
   files out of Git.
5. Run resource-filtered candidate discovery:

   ```powershell
   python .\01_code\novel_candidate_score.py
   ```

   This stage requires the strict experiment outputs plus CIViC, IntOGen/OncoPlex,
   and PCAWG reference files listed in the input manifest. Its outputs contain
   scores and feature values and are intentionally not part of the release.
6. Compare cohort predictions after preparing four CSV/TSV files with `Gene`
   and `Score` columns. Predictions at or above 0.50 are counted as drivers:

   ```powershell
   python .\01_code\gene_prediction_overlap.py `
     --tcga .\predictions\tcga.csv `
     --ccrcc .\predictions\ccrcc.csv `
     --hnscc .\predictions\hnscc.csv `
     --pcawg .\predictions\pcawg.csv `
     --output-dir .\prediction_overlap
   ```

   The output contains per-gene membership and pairwise overlap summaries.
7. For PCAWG analyses, list tasks with
   `python .\05_external_validation\pcawg_validation.py --help`. For CPTAC
   input download and feature preparation, use
   `python .\05_external_validation\cptac_data.py --help`. For CPTAC
   validation, use `python .\05_external_validation\cptac_validation.py --help`.
   Download PCAWG metadata and inventory with
   `.\05_external_validation\code\pcawg_data.ps1 -Action Inventory`, then
   download its large core inputs with
   `.\05_external_validation\code\pcawg_data.ps1 -Action CoreData`. Download
   CPTAC files with `cptac_data.py download`. Required files are listed in
   `02_data/manifests/input_manifest.csv`. External feature tables and
   validation results are not bundled.

## Scope and rerun notes

- The existing feature extraction scripts are retained from the research
  pipeline and use fixed input filenames. Follow the manifest and working
  directory instructions; do not assume that running a script downloads every
  required file automatically.
- The paper's exact end-to-end run cannot be rerun from this repository
  alone until the project-specific base matrix/label ledger and licensed or
  controlled-access inputs are supplied.
- PCAWG/CPTAC validation code is included as source, but its calculations are
  not asserted as verified by this repository-only build. Re-run the relevant
  scripts after staging their external inputs and inspect their generated
  manifests.
- No raw data or feature matrices were copied into this release. If you later
  publish this repository, review every provider's terms and the final Git
  diff before pushing.
- Choose a license for the authors' code and confirm data-provider terms
  before making the repository public. This folder has not been pushed to
  GitHub. After installing Git, initialize locally with `git init -b main`,
  inspect `git status` and the ignore rules, then commit/push only after review.
