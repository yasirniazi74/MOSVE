from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _run_audit(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_audit.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import json
        from pathlib import Path
        import pandas as pd
        WORK_DIR = Path(__file__).resolve().parents[2]
        ROOT = WORK_DIR / '05_external_validation'
        RAW = ROOT / '02_raw_data'
        PROCESSED = ROOT / '03_processed_data'
        LABELS = ROOT / '05_labels'
        TCGA_PROJECTS = {'BLCA-US', 'BRCA-US', 'CESC-US', 'COAD-US', 'DLBC-US', 'GBM-US', 'HNSC-US', 'KICH-US', 'KIRC-US', 'KIRP-US', 'LAML-US', 'LGG-US', 'LIHC-US', 'LUAD-US', 'LUSC-US', 'OV-US', 'PAAD-US', 'PRAD-US', 'READ-US', 'SARC-US', 'SKCM-US', 'STAD-US', 'THCA-US', 'UCEC-US'}

        def main() -> None:
            PROCESSED.mkdir(parents=True, exist_ok=True)
            LABELS.mkdir(parents=True, exist_ok=True)
            samples = pd.read_csv(RAW / 'metadata' / 'pcawg_sample_sheet.tsv', sep='\t')
            samples['dcc_project_code'] = samples['dcc_project_code'].astype(str).str.strip()
            samples['is_tcga_project'] = samples['dcc_project_code'].isin(TCGA_PROJECTS)
            samples['is_tumour'] = samples['dcc_specimen_type'].astype(str).str.lower().str.contains('tumou|tumor')
            samples['is_whitelist'] = samples['donor_wgs_exclusion_white_gray'].eq('Whitelist')
            non_tcga = samples.loc[~samples['is_tcga_project']].copy()
            eligible = non_tcga.loc[non_tcga['is_whitelist'] & non_tcga['is_tumour']].copy()
            eligible.to_csv(PROCESSED / 'non_tcga_pcawg_eligible_samples.csv', index=False)
            project_rows = []
            for project, group in non_tcga.groupby('dcc_project_code', sort=True):
                eligible_group = eligible.loc[eligible['dcc_project_code'].eq(project)]
                project_rows.append({'Project': project, 'All_Donors': group['donor_unique_id'].nunique(), 'Eligible_Tumour_Donors': eligible_group['donor_unique_id'].nunique(), 'Eligible_Tumour_Samples': eligible_group['icgc_sample_id'].nunique(), 'WGS_Samples': eligible_group.loc[eligible_group['library_strategy'].eq('WGS'), 'icgc_sample_id'].nunique(), 'RNA_Seq_Samples': eligible_group.loc[eligible_group['library_strategy'].eq('RNA-Seq'), 'icgc_sample_id'].nunique(), 'Library_Strategies': '|'.join(sorted(eligible_group['library_strategy'].dropna().astype(str).unique()))})
            projects = pd.DataFrame(project_rows).sort_values(['Eligible_Tumour_Donors', 'Project'], ascending=[False, True])
            projects.to_csv(PROCESSED / 'non_tcga_pcawg_project_coverage.csv', index=False)
            driver_path = RAW / 'driver_labels' / 'TableS1_compendium_mutational_drivers.xlsx'
            drivers = pd.read_excel(driver_path, sheet_name='TableS1_compendium_mutational_drivers_Jan12_2018')
            drivers['Gene'] = drivers['Gene'].astype(str).str.strip().str.upper()
            drivers = drivers.loc[drivers['Gene'].ne('') & drivers['Gene'].ne('NAN')].copy()
            drivers.to_csv(LABELS / 'pcawg_consensus_driver_elements.csv', index=False)
            driver_genes = drivers[['Gene']].drop_duplicates().sort_values('Gene').reset_index(drop=True)
            driver_genes.to_csv(LABELS / 'pcawg_consensus_driver_genes.csv', index=False)
            summary = {'all_projects': int(samples['dcc_project_code'].nunique()), 'tcga_projects_present': int(samples.loc[samples['is_tcga_project'], 'dcc_project_code'].nunique()), 'non_tcga_projects': int(non_tcga['dcc_project_code'].nunique()), 'all_donors': int(samples['donor_unique_id'].nunique()), 'non_tcga_donors': int(non_tcga['donor_unique_id'].nunique()), 'eligible_non_tcga_tumour_donors': int(eligible['donor_unique_id'].nunique()), 'eligible_non_tcga_tumour_samples': int(eligible['icgc_sample_id'].nunique()), 'eligible_non_tcga_wgs_samples': int(eligible.loc[eligible['library_strategy'].eq('WGS'), 'icgc_sample_id'].nunique()), 'eligible_non_tcga_rna_seq_samples': int(eligible.loc[eligible['library_strategy'].eq('RNA-Seq'), 'icgc_sample_id'].nunique()), 'pcawg_driver_element_rows': int(len(drivers)), 'pcawg_unique_driver_genes': int(driver_genes['Gene'].nunique()), 'tcga_project_codes_excluded': sorted(TCGA_PROJECTS)}
            (PROCESSED / 'pcawg_metadata_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary, indent=2))
            print('\nTop non-TCGA projects by eligible tumour donors:')
            print(projects.head(15).to_string(index=False))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_build(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_features.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import json
        import re
        from collections import defaultdict
        from pathlib import Path
        import numpy as np
        import pandas as pd
        WORK_DIR = Path(__file__).resolve().parents[2]
        ROOT = WORK_DIR / '05_external_validation'
        RAW = ROOT / '02_raw_data'
        PROCESSED = ROOT / '03_processed_data'
        FEATURES = ROOT / '04_features'
        INTERNAL_MATRIX = WORK_DIR / '04_strict_verified_4420_experiment' / 'data' / 'strict_verified_4420_feature_matrix.csv'
        TCGA_PROJECTS = {'BLCA-US', 'BRCA-US', 'CESC-US', 'COAD-US', 'DLBC-US', 'GBM-US', 'HNSC-US', 'KICH-US', 'KIRC-US', 'KIRP-US', 'LAML-US', 'LGG-US', 'LIHC-US', 'LUAD-US', 'LUSC-US', 'OV-US', 'PAAD-US', 'PRAD-US', 'READ-US', 'SARC-US', 'SKCM-US', 'STAD-US', 'THCA-US', 'UCEC-US'}

        def eligible_aliquots() -> tuple[set[str], set[str]]:
            samples = pd.read_csv(RAW / 'metadata' / 'pcawg_sample_sheet.tsv', sep='\t')
            tumour = samples['dcc_specimen_type'].astype(str).str.lower().str.contains('tumou|tumor')
            eligible = samples.loc[~samples['dcc_project_code'].isin(TCGA_PROJECTS) & samples['donor_wgs_exclusion_white_gray'].eq('Whitelist') & tumour].copy()
            wgs = set(eligible.loc[eligible['library_strategy'].eq('WGS'), 'aliquot_id'].dropna().astype(str))
            rna = set(eligible.loc[eligible['library_strategy'].eq('RNA-Seq'), 'aliquot_id'].dropna().astype(str))
            return (wgs, rna)

        def build_mutation_features(gene_universe: set[str], eligible_wgs: set[str]) -> pd.DataFrame:
            path = RAW / 'mutation' / 'final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
            usecols = ['Hugo_Symbol', 'Variant_Classification', 'Variant_Type', 'Tumor_Sample_Barcode', 'Project_Code', 'Start_position', 'i_VAF']
            counts = defaultdict(lambda: defaultdict(int))
            patients: dict[str, set[str]] = defaultdict(set)
            positions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            vaf_sum = defaultdict(float)
            vaf_n = defaultdict(int)
            total_rows = retained_rows = 0
            for chunk in pd.read_csv(path, sep='\t', compression='gzip', usecols=usecols, chunksize=250000, low_memory=False):
                total_rows += len(chunk)
                chunk['Hugo_Symbol'] = chunk['Hugo_Symbol'].astype(str).str.strip().str.upper()
                keep = ~chunk['Project_Code'].isin(TCGA_PROJECTS) & chunk['Tumor_Sample_Barcode'].astype(str).isin(eligible_wgs) & chunk['Hugo_Symbol'].isin(gene_universe)
                chunk = chunk.loc[keep].copy()
                retained_rows += len(chunk)
                if chunk.empty:
                    continue
                for gene, group in chunk.groupby('Hugo_Symbol', sort=False):
                    vc = group['Variant_Classification'].fillna('').astype(str)
                    c = counts[gene]
                    c['SNV_Count'] += len(group)
                    c['Missense'] += int(vc.eq('Missense_Mutation').sum())
                    c['Nonsense'] += int(vc.eq('Nonsense_Mutation').sum())
                    c['Frameshift'] += int(vc.str.contains('Frame_Shift', regex=False).sum())
                    c['Silent'] += int(vc.eq('Silent').sum())
                    c['Splice_Site'] += int(vc.eq('Splice_Site').sum())
                    patients[gene].update(group['Tumor_Sample_Barcode'].dropna().astype(str))
                    for position, n in group['Start_position'].value_counts().items():
                        positions[gene][str(position)] += int(n)
                    vaf = pd.to_numeric(group['i_VAF'], errors='coerce').dropna()
                    vaf_sum[gene] += float(vaf.sum())
                    vaf_n[gene] += int(vaf.count())
            rows = []
            for gene in sorted(gene_universe):
                c = counts[gene]
                total = int(c['SNV_Count'])
                missense = int(c['Missense'])
                nonsense = int(c['Nonsense'])
                frameshift = int(c['Frameshift'])
                silent = int(c['Silent'])
                splice = int(c['Splice_Site'])
                non_silent = missense + nonsense + frameshift + splice
                truncating = nonsense + frameshift
                position_counts = list(positions[gene].values())
                recurrent = sum((value >= 2 for value in position_counts))
                rows.append({'Gene': gene, 'SNV_Count': total, 'Missense': missense, 'Nonsense': nonsense, 'Frameshift': frameshift, 'Silent': silent, 'Splice_Site': splice, 'Patients_Affected': len(patients[gene]), 'NonSilent_Ratio': non_silent / max(total, 1), 'Truncating_Ratio': truncating / max(total, 1), 'Pathogenicity_Score': truncating + non_silent, 'High_Impact_Ratio': truncating / max(total, 1), 'Missense_Ratio': missense / max(total, 1), 'Silent_Ratio': silent / max(total, 1), 'Mutation_Impact_Score': 2 * truncating + non_silent, 'Hotspot_Max_Same_Position': max(position_counts, default=0), 'Hotspot_Recurrent_Position_Count': recurrent, 'Hotspot_Fraction': recurrent / max(len(position_counts), 1), 'Mean_Tumor_VAF': vaf_sum[gene] / max(vaf_n[gene], 1)})
            result = pd.DataFrame(rows)
            result.attrs['total_maf_rows'] = total_rows
            result.attrs['retained_maf_rows'] = retained_rows
            return result

        def build_cnv_features(gene_universe: set[str], eligible_wgs: set[str]) -> tuple[pd.DataFrame, int]:
            absolute_path = RAW / 'cnv' / 'all_samples.consensus_CN.by_gene.170214.txt.gz'
            levels_path = RAW / 'cnv' / 'all_samples.consensus_level_calls.by_gene.170214.txt.gz'
            header = pd.read_csv(absolute_path, sep='\t', compression='gzip', nrows=0).columns.tolist()
            sample_columns = [column for column in header[3:] if column in eligible_wgs]
            usecols = header[:3] + sample_columns
            absolute = pd.read_csv(absolute_path, sep='\t', compression='gzip', usecols=usecols)
            levels = pd.read_csv(levels_path, sep='\t', compression='gzip', usecols=usecols)
            absolute['Gene'] = absolute['Gene Symbol'].astype(str).str.split('|').str[0].str.upper()
            levels['Gene'] = levels['Gene Symbol'].astype(str).str.split('|').str[0].str.upper()
            absolute = absolute.loc[absolute['Gene'].isin(gene_universe)].drop_duplicates('Gene')
            levels = levels.loc[levels['Gene'].isin(gene_universe)].drop_duplicates('Gene')
            abs_values = absolute[sample_columns].apply(pd.to_numeric, errors='coerce')
            level_values = levels[sample_columns].apply(pd.to_numeric, errors='coerce')
            output = pd.DataFrame({'Gene': absolute['Gene'].values, 'CNV_Mean': abs_values.mean(axis=1).values, 'CNV_STD': abs_values.std(axis=1).values, 'CNV_Gain_Ratio': (level_values > 0).mean(axis=1).values, 'CNV_Loss_Ratio': (level_values < 0).mean(axis=1).values, 'CNV_Amplification_Ratio': (level_values >= 1).mean(axis=1).values, 'CNV_Deletion_Ratio': (level_values <= -1).mean(axis=1).values})
            return (output, len(sample_columns))

        def parse_gencode_map() -> dict[str, str]:
            gtf = RAW / 'metadata' / 'rnaseq.gc19_extNc.gtf'
            pattern = re.compile('gene_id "gencode::([^:]+)::tc_(ENSG\\d+)(?:\\.\\d+)?"')
            mapping: dict[str, str] = {}
            with gtf.open('r', encoding='utf-8') as handle:
                for line in handle:
                    match = pattern.search(line)
                    if match:
                        symbol, ensembl = match.groups()
                        mapping.setdefault(ensembl, symbol.upper())
            return mapping

        def build_expression_features(gene_universe: set[str], eligible_rna: set[str]) -> tuple[pd.DataFrame, int, int]:
            path = RAW / 'expression' / 'tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz'
            header = pd.read_csv(path, sep='\t', compression='gzip', nrows=0).columns.tolist()
            sample_columns = [column for column in header[1:] if column in eligible_rna]
            expression = pd.read_csv(path, sep='\t', compression='gzip', usecols=['feature'] + sample_columns)
            mapping = parse_gencode_map()
            expression['Ensembl'] = expression['feature'].astype(str).str.split('.').str[0]
            expression['Gene'] = expression['Ensembl'].map(mapping)
            expression = expression.loc[expression['Gene'].isin(gene_universe)].copy()
            values = expression[sample_columns].apply(pd.to_numeric, errors='coerce')
            expression['Expr_Mean'] = values.mean(axis=1)
            expression['Expr_STD'] = values.std(axis=1)
            expression['Expr_Median'] = values.median(axis=1)
            expression['Expr_CV'] = expression['Expr_STD'] / (expression['Expr_Mean'].abs() + 1e-06)
            result = expression.groupby('Gene', as_index=False)[['Expr_Mean', 'Expr_STD', 'Expr_Median', 'Expr_CV']].mean()
            return (result, len(sample_columns), len(mapping))

        def main() -> None:
            FEATURES.mkdir(parents=True, exist_ok=True)
            base = pd.read_csv(INTERNAL_MATRIX, usecols=['Gene'])
            gene_universe = set(base['Gene'].astype(str).str.strip().str.upper())
            eligible_wgs, eligible_rna = eligible_aliquots()
            print('Building mutation features ...', flush=True)
            mutation = build_mutation_features(gene_universe, eligible_wgs)
            mutation_total = mutation.attrs['total_maf_rows']
            mutation_retained = mutation.attrs['retained_maf_rows']
            mutation.to_csv(FEATURES / 'pcawg_non_tcga_mutation_features.csv', index=False)
            print('Building CNV features ...', flush=True)
            cnv, cnv_samples = build_cnv_features(gene_universe, eligible_wgs)
            cnv.to_csv(FEATURES / 'pcawg_non_tcga_cnv_features.csv', index=False)
            print('Building expression features ...', flush=True)
            expression, expression_samples, mapped_ensembl = build_expression_features(gene_universe, eligible_rna)
            expression.to_csv(FEATURES / 'pcawg_non_tcga_expression_features.csv', index=False)
            combined = base.copy()
            combined['Gene'] = combined['Gene'].astype(str).str.strip().str.upper()
            for frame in [mutation, cnv, expression]:
                combined = combined.merge(frame, on='Gene', how='left')
            combined.to_csv(FEATURES / 'pcawg_non_tcga_dynamic_gene_features.csv', index=False)
            summary = {'gene_universe': len(gene_universe), 'eligible_wgs_aliquots': len(eligible_wgs), 'eligible_rna_aliquots': len(eligible_rna), 'maf_rows_scanned': mutation_total, 'non_tcga_universe_maf_rows': mutation_retained, 'genes_with_mutations': int(mutation['SNV_Count'].gt(0).sum()), 'cnv_samples_matched': cnv_samples, 'genes_with_cnv': int(cnv['CNV_Mean'].notna().sum()), 'expression_samples_matched': expression_samples, 'gencode_ensembl_mappings': mapped_ensembl, 'genes_with_expression': int(expression['Expr_Mean'].notna().sum()), 'methylation_available': False}
            (FEATURES / 'pcawg_feature_build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary, indent=2))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_methylation(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_methylation.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import csv
        import gzip
        import json
        from collections import defaultdict
        from pathlib import Path
        import numpy as np
        import pandas as pd
        WORK_DIR = Path(__file__).resolve().parents[2]
        EXT = WORK_DIR / '05_external_validation'
        METH_DIR = EXT / '02_raw_data' / 'methylation' / 'PACA-AU'
        ANNOTATION = WORK_DIR / '02_data' / 'raw_sources' / 'humanmethylation450_15017482_v1-2.csv'
        INTERNAL = WORK_DIR / '04_strict_verified_4420_experiment' / 'data' / 'strict_verified_4420_feature_matrix.csv'
        OUT_DIR = EXT / '04_features'
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT = OUT_DIR / 'pcawg_paca_au_methylation_features.csv'
        genes = set(pd.read_csv(INTERNAL, usecols=['Gene'])['Gene'].astype(str).str.upper())
        ann = pd.read_csv(ANNOTATION, skiprows=7, usecols=['IlmnID', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group'], low_memory=False)
        probe_map = {}
        for row in ann.itertuples(index=False):
            probe = str(row.IlmnID).strip()
            names = str(row.UCSC_RefGene_Name) if pd.notna(row.UCSC_RefGene_Name) else ''
            groups = str(row.UCSC_RefGene_Group) if pd.notna(row.UCSC_RefGene_Group) else ''
            mapped = []
            for i, name in enumerate(names.split(';')):
                gene = name.strip().upper()
                if gene in genes:
                    group = groups.split(';')[i].strip().lower() if i < len(groups.split(';')) else ''
                    mapped.append((gene, group))
            if mapped:
                probe_map[probe] = mapped
        stats = defaultdict(lambda: [0.0, 0.0, 0, 0, 0.0, 0.0, 0])
        seen = set()
        files = sorted(METH_DIR.rglob('*.gz'))
        rows = 0
        mapped_rows = 0
        for path in files:
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace', newline='') as fh:
                reader = csv.reader(fh, delimiter='\t')
                for fields in reader:
                    if len(fields) < 10:
                        continue
                    sample = fields[3].strip()
                    probe = fields[7].strip()
                    if not sample or probe not in probe_map:
                        continue
                    key = (sample, probe)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        beta = float(fields[8])
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(beta) or beta < 0 or beta > 1:
                        continue
                    rows += 1
                    for gene, group in probe_map[probe]:
                        s = stats[gene]
                        s[0] += beta
                        s[1] += beta * beta
                        s[2] += 1
                        s[3] += int(beta >= 0.7)
                        s[4] += beta if any((x in group for x in ('tss200', 'tss1500', '1stexon', "5'utr"))) else 0.0
                        s[5] += int(any((x in group for x in ('tss200', 'tss1500', '1stexon', "5'utr"))))
                        s[6] += int(beta <= 0.3)
                    mapped_rows += 1
        out = pd.DataFrame({'Gene': sorted(genes)})
        records = []
        for gene in out['Gene']:
            total, sq, n, hyper, prom_sum, prom_n, hypo = stats[gene]
            mean = total / n if n else np.nan
            std = max(sq / n - mean * mean, 0.0) ** 0.5 if n else np.nan
            records.append({'Gene': gene, 'Meth_Mean': mean, 'Meth_Std': std, 'Meth_Probe_Count': n, 'Meth_Hyper_Ratio': hyper / n if n else np.nan, 'Meth_Hypo_Ratio': hypo / n if n else np.nan, 'Prom_Meth_Mean': prom_sum / prom_n if prom_n else np.nan, 'Prom_Meth_Probe_Count': prom_n})
        pd.DataFrame(records).to_csv(OUT, index=False)
        summary = {'project': 'PACA-AU', 'gene_universe': len(genes), 'annotation_probes_mapped': len(probe_map), 'methylation_files': len(files), 'unique_sample_probe_values': len(seen), 'valid_mapped_values': rows, 'mapped_rows': mapped_rows, 'genes_with_methylation': int((out.set_index('Gene').index.map(lambda g: stats[g][2]) > 0).sum()), 'output': str(OUT)}
        (OUT_DIR / 'pcawg_paca_au_methylation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    finally:
        _sys.argv = _runner_previous_argv

def _run_merge(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_merge.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        import json
        import pandas as pd
        ROOT = Path(__file__).resolve().parents[2]
        EXT = ROOT / '05_external_validation'
        features = EXT / '04_features'
        base = pd.read_csv(features / 'pcawg_non_tcga_dynamic_gene_features.csv')
        meth = pd.read_csv(features / 'pcawg_paca_au_methylation_features.csv')
        internal = pd.read_csv(ROOT / '04_strict_verified_4420_experiment' / 'data' / 'strict_verified_4420_feature_matrix.csv')
        base['Gene'] = base['Gene'].astype(str).str.upper()
        meth['Gene'] = meth['Gene'].astype(str).str.upper()
        out = base.drop(columns=[c for c in meth.columns if c != 'Gene' and c in base.columns], errors='ignore')
        out = out.merge(meth, on='Gene', how='left', validate='one_to_one')
        static = internal.drop(columns=[c for c in out.columns if c != 'Gene' and c in internal.columns], errors='ignore')
        out = static.merge(out, on='Gene', how='left', validate='one_to_one')
        out = out.sort_values('Gene').reset_index(drop=True)
        dest = features / 'pcawg_non_tcga_paca_au_external_features.csv'
        out.to_csv(dest, index=False)
        summary = {'genes': len(out), 'features': len(out.columns) - 1, 'methylation_columns': [c for c in meth.columns if c != 'Gene'], 'genes_with_meth_mean': int(out['Meth_Mean'].notna().sum()), 'output': str(dest)}
        (features / 'pcawg_external_feature_merge_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
    finally:
        _sys.argv = _runner_previous_argv

def _run_external_split(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_external_split.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        import json
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        ROOT = Path(__file__).resolve().parents[2]
        INP = ROOT / '05_external_validation/07_unseen_gene_test/ratio_matched_591'
        OUT = ROOT / '05_external_validation/21_pcawg_external_only_gene_split'
        OUT.mkdir(parents=True, exist_ok=True)
        SEED = 42
        THRESHOLD = 0.5

        def pipe(m, scale=False):
            s = [('imputer', SimpleImputer(strategy='median'))]
            if scale:
                s.append(('scaler', StandardScaler()))
            s.append(('model', m))
            return Pipeline(s)

        def model():
            return VotingClassifier(estimators=[('ExtraTrees', pipe(ExtraTreesClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=0.5, class_weight='balanced', random_state=SEED, n_jobs=1))), ('HistGradientBoosting', pipe(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED))), ('LogisticRegression', pipe(LogisticRegression(C=0.06951927961775606, solver='liblinear', penalty='l2', max_iter=7000, class_weight='balanced', random_state=SEED), True)), ('LightGBM', pipe(LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=0.0, random_state=SEED, n_jobs=1, verbose=-1)))], voting='soft', n_jobs=1)

        def met(y, s):
            p = (s >= THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            return {'N': len(y), 'Positive': int(y.sum()), 'Negative': int((1 - y).sum()), 'ROC_AUC': roc_auc_score(y, s), 'AUPRC': average_precision_score(y, s), 'Accuracy': accuracy_score(y, p), 'Precision': precision_score(y, p, zero_division=0), 'Recall': recall_score(y, p), 'F1': f1_score(y, p), 'MCC': matthews_corrcoef(y, p), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

        def main():
            x = pd.read_csv(INP / 'external_gene_scores.csv')
            selected = pd.read_csv(ROOT / '05_external_validation/07_unseen_gene_test/selected_features.csv').Feature.tolist()
            if x.Gene.nunique() != 591 or x.External_Label.sum() != 226:
                raise RuntimeError('Expected the ratio-matched 591-gene external set.')
            if set(selected) - set(x):
                raise RuntimeError('Stored external matrix lacks portable features.')
            train, tmp = train_test_split(x, test_size=0.4, stratify=x.External_Label, random_state=SEED)
            val, test = train_test_split(tmp, test_size=0.5, stratify=tmp.External_Label, random_state=SEED)
            clf = model().fit(train[selected], train.External_Label)
            rows = []
            for split, part in [('Validation', val), ('Test', test)]:
                score = clf.predict_proba(part[selected])[:, 1]
                rows.append({'Split': split, 'Algorithm': 'MOSVE ensemble', **met(part.External_Label.to_numpy(), score)})
                p = part[['Gene', 'External_Label', 'Reference_Class']].copy()
                p['MOSVE_Score'] = score
                p['Predicted_Driver_0_50'] = (score >= THRESHOLD).astype(int)
                p.to_csv(OUT / f'{split.lower()}_predictions.csv', index=False)
            for name, est in clf.named_estimators_.items():
                score = est.predict_proba(test[selected])[:, 1]
                rows.append({'Split': 'Test', 'Algorithm': name, **met(test.External_Label.to_numpy(), score)})
            for split, part in [('train', train), ('validation', val), ('test', test)]:
                part[['Gene', 'External_Label', 'Reference_Class']].to_csv(OUT / f'{split}_genes.csv', index=False)
            pd.DataFrame(rows).to_csv(OUT / 'metrics.csv', index=False)
            (OUT / 'manifest.json').write_text(json.dumps({'dataset': '591-gene ratio-matched PCAWG external universe only', 'training_universe_overlap_with_MOSVE_4420': 0, 'features': len(selected), 'threshold': THRESHOLD, 'split': 'stratified random 60/20/20 by gene', 'train': {'genes': len(train), 'positives': int(train.External_Label.sum())}, 'validation': {'genes': len(val), 'positives': int(val.External_Label.sum())}, 'test': {'genes': len(test), 'positives': int(test.External_Label.sum())}, 'labels': 'PCAWG consensus-driver positives; matched reference background controls'}, indent=2))
            print(pd.DataFrame(rows).round(4).to_string(index=False))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_expand(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_expand.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        from collections import Counter, defaultdict
        import json
        import numpy as np
        import pandas as pd
        ROOT = Path(__file__).resolve().parents[2]
        RAW = ROOT / '05_external_validation/02_raw_data'
        BASE = ROOT / '05_external_validation/04_features/pcawg_non_tcga_paca_au_external_features.csv'
        TCGA = ROOT / '04_strict_verified_4420_experiment/data/strict_verified_4420_feature_matrix.csv'
        OUT = ROOT / '05_external_validation/25_pcawg_expanded_dynamic_features'
        OUT.mkdir(parents=True, exist_ok=True)
        TCGA_PROJECTS = {'BLCA-US', 'BRCA-US', 'CESC-US', 'COAD-US', 'DLBC-US', 'GBM-US', 'HNSC-US', 'KICH-US', 'KIRC-US', 'KIRP-US', 'LAML-US', 'LGG-US', 'LIHC-US', 'LUAD-US', 'LUSC-US', 'OV-US', 'PAAD-US', 'PRAD-US', 'READ-US', 'SARC-US', 'SKCM-US', 'STAD-US', 'THCA-US', 'UCEC-US'}

        def main():
            d = pd.read_csv(BASE)
            genes = set(d.Gene.astype(str).str.upper())
            d.Gene = d.Gene.astype(str).str.upper()
            s = pd.read_csv(RAW / 'metadata/pcawg_sample_sheet.tsv', sep='\t')
            ok = ~s.dcc_project_code.isin(TCGA_PROJECTS) & s.donor_wgs_exclusion_white_gray.eq('Whitelist') & s.dcc_specimen_type.astype(str).str.lower().str.contains('tumou|tumor') & s.library_strategy.eq('WGS')
            aliquots = set(s.loc[ok, 'aliquot_id'].dropna().astype(str))
            maf = RAW / 'mutation/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
            use = ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Project_Code', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2', 'ref_context', 'i_VAF', 'Start_position']
            n = Counter()
            snp = Counter()
            indel = Counter()
            cpg = Counter()
            ct = Counter()
            vsum = Counter()
            vn = Counter()
            vmax = defaultdict(lambda: np.nan)
            ctx = defaultdict(Counter)
            pos = defaultdict(Counter)
            for q in pd.read_csv(maf, sep='\t', compression='gzip', usecols=use, chunksize=250000, low_memory=False):
                q.Hugo_Symbol = q.Hugo_Symbol.astype(str).str.upper()
                q = q[~q.Project_Code.isin(TCGA_PROJECTS) & q.Tumor_Sample_Barcode.astype(str).isin(aliquots) & q.Hugo_Symbol.isin(genes)]
                for g, x in q.groupby('Hugo_Symbol'):
                    n[g] += len(x)
                    snp[g] += int(x.Variant_Type.eq('SNP').sum())
                    indel[g] += int(x.Variant_Type.isin(['INS', 'DEL']).sum())
                    cpg[g] += int(x.ref_context.astype(str).str.upper().str.contains('CG').sum())
                    ct[g] += int((x.Reference_Allele.eq('C') & x.Tumor_Seq_Allele2.eq('T') | x.Reference_Allele.eq('G') & x.Tumor_Seq_Allele2.eq('A')).sum())
                    ctx[g].update(x.ref_context.dropna().astype(str).str.upper())
                    pos[g].update(x.Start_position.dropna().astype(str))
                    v = pd.to_numeric(x.i_VAF, errors='coerce').dropna()
                    vsum[g] += float(v.sum())
                    vn[g] += len(v)
                    vmax[g] = np.nanmax([vmax[g], v.max()]) if len(v) else vmax[g]
            rows = []
            for g in d.Gene:
                total = n[g]
                p = np.array(list(ctx[g].values()), float)
                p = p / p.sum() if p.sum() else p
                entropy = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
                recurrent = sum((v >= 2 for v in pos[g].values()))
                rows.append({'Gene': g, 'Max_Tumor_VAF': vmax[g] if not np.isnan(vmax[g]) else 0.0, 'Mutation_Context_Entropy': entropy, 'CpG_Context_Ratio': cpg[g] / max(total, 1), 'C_to_T_Context_Ratio': ct[g] / max(total, 1), 'SNP_Fraction': snp[g] / max(total, 1), 'Indel_Fraction': indel[g] / max(total, 1), 'Protein_Position_Cluster_Score': recurrent / max(len(pos[g]), 1)})
            extra = pd.DataFrame(rows)
            out = d.drop(columns=[c for c in extra if c != 'Gene'], errors='ignore').merge(extra, on='Gene', how='left')
            out['HighImpact_to_Context_Ratio'] = out['High_Impact_Ratio'] * out['Mutation_Context_Entropy']
            bases = ['SNV_Count', 'Missense', 'Nonsense', 'Frameshift', 'Silent', 'Splice_Site', 'Patients_Affected', 'Hotspot_Max_Same_Position', 'Hotspot_Recurrent_Position_Count']
            for b in bases:
                if b in out:
                    if 'FE_log1p_' + b in out:
                        out['FE_log1p_' + b] = np.log1p(pd.to_numeric(out[b], errors='coerce'))
                    if 'FE_sqrt_' + b in out:
                        out['FE_sqrt_' + b] = np.sqrt(pd.to_numeric(out[b], errors='coerce').clip(lower=0))
            out.to_csv(OUT / 'pcawg_expanded_reproducible_dynamic_4420.csv', index=False)
            t = pd.read_csv(TCGA).set_index('Gene')
            o = out.set_index('Gene')
            features = [c for c in t if c != 'Label' and c in o]
            changed = []
            for c in features:
                a = pd.to_numeric(t[c], errors='coerce')
                b = pd.to_numeric(o[c], errors='coerce')
                m = a.notna() & b.notna()
                changed.append({'Feature': c, 'Both_Observed': int(m.sum()), 'Values_Changed': bool(m.any() and (not np.isclose(a[m], b[m], rtol=1e-10, atol=1e-12).all()))})
            audit = pd.DataFrame(changed)
            audit.to_csv(OUT / 'tcga_vs_expanded_pcawg_change_audit.csv', index=False)
            summary = {'features_total': len(features), 'features_changed': int(audit.Values_Changed.sum()), 'raw_maf_aliquots': len(aliquots), 'recomputed_features': list(extra.columns[1:]) + ['HighImpact_to_Context_Ratio'], 'note': 'Protein_Position_Cluster_Score is a documented genomic-position recurrence proxy because PCAWG MAF lacks protein-position annotation.'}
            (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_cv10(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/pcawg_cv10.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        ROOT = Path(__file__).resolve().parents[2]
        INP = ROOT / '05_external_validation/07_unseen_gene_test/ratio_matched_591'
        OUT = ROOT / '05_external_validation/27_pcawg_external591_cv10'
        OUT.mkdir(parents=True, exist_ok=True)
        SEED = 42
        T = 0.5

        def pipe(m, s=False):
            z = [('imputer', SimpleImputer(strategy='median'))]
            if s:
                z.append(('scaler', StandardScaler()))
            z.append(('model', m))
            return Pipeline(z)

        def build():
            return VotingClassifier(estimators=[('ExtraTrees', pipe(ExtraTreesClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=0.5, class_weight='balanced', random_state=SEED, n_jobs=1))), ('HistGradientBoosting', pipe(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED))), ('LogisticRegression', pipe(LogisticRegression(C=0.06951927961775606, solver='liblinear', penalty='l2', max_iter=7000, class_weight='balanced', random_state=SEED), True)), ('LightGBM', pipe(LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=0.0, random_state=SEED, n_jobs=1, verbose=-1)))], voting='soft', n_jobs=1)

        def met(y, s):
            p = (s >= T).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            return {'ROC_AUC': roc_auc_score(y, s), 'AUPRC': average_precision_score(y, s), 'Accuracy': accuracy_score(y, p), 'Precision': precision_score(y, p, zero_division=0), 'Recall': recall_score(y, p), 'F1': f1_score(y, p), 'MCC': matthews_corrcoef(y, p), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

        def main():
            d = pd.read_csv(INP / 'external_gene_scores.csv')
            features = pd.read_csv(ROOT / '05_external_validation/07_unseen_gene_test/selected_features.csv').Feature.tolist()
            y = d.External_Label.to_numpy()
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
            names = ['ExtraTrees', 'HistGradientBoosting', 'LogisticRegression', 'LightGBM', 'MOSVE ensemble']
            oof = {n: np.full(len(d), np.nan) for n in names}
            foldrows = []
            for fold, (tr, te) in enumerate(skf.split(d, y), 1):
                m = build().fit(d.iloc[tr][features], y[tr])
                parts = list(m.named_estimators_.items()) + [('MOSVE ensemble', m)]
                for n, e in parts:
                    s = e.predict_proba(d.iloc[te][features])[:, 1]
                    oof[n][te] = s
                    foldrows.append({'Fold': fold, 'Model': n, 'N': len(te), 'Positive': int(y[te].sum()), **met(y[te], s)})
            rows = []
            for n, s in oof.items():
                rows.append({'Model': n, **met(y, s)})
            pd.DataFrame(rows).to_csv(OUT / 'oof_metrics.csv', index=False)
            pd.DataFrame(foldrows).to_csv(OUT / 'fold_metrics.csv', index=False)
            pred = d[['Gene', 'External_Label', 'Reference_Class']].copy()
            for n, s in oof.items():
                pred[n.replace(' ', '_') + '_OOF_Score'] = s
            pred.to_csv(OUT / 'oof_predictions.csv', index=False)
            (OUT / 'manifest.json').write_text(json.dumps({'dataset': '591-gene PCAWG external universe only', 'features': len(features), 'cv': 'stratified 10-fold, shuffled seed 42', 'threshold': T, 'labels': '226 PCAWG positives and 365 matched reference-background controls', 'mosve_4420_gene_overlap': 0}, indent=2))
            print(pd.DataFrame(rows).round(4).to_string(index=False))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['audit', 'build', 'methylation', 'merge', 'external_split', 'expand', 'cv10'])
    arguments = _sys.argv[1:]
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser.print_help()
        return
    args = parser.parse_args(arguments[:1])
    forwarded = arguments[1:]
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if forwarded:
        if forwarded in (['-h'], ['--help']):
            parser.print_help()
            return
        parser.error(f'{args.task} does not accept extra arguments')
    {'audit': _run_audit, 'build': _run_build, 'methylation': _run_methylation, 'merge': _run_merge, 'external_split': _run_external_split, 'expand': _run_expand, 'cv10': _run_cv10}[args.task]([])

if __name__ == '__main__':
    main()
