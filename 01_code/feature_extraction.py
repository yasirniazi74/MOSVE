from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _run_sequences(forwarded_args):
    __file__ = str(ROOT / '01_code/_runtime/feature_sequences.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import os, time, shutil, tempfile, warnings, json, re
        import urllib.parse, urllib.request
        import xml.etree.ElementTree as ET
        from collections import Counter, defaultdict
        from math import log2
        import pandas as pd
        import numpy as np
        warnings.filterwarnings('ignore')
        REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        RAW_SOURCE_DIR = os.path.join(REPO_ROOT, '02_data', 'raw_sources')
        LABEL_SOURCE_DIR = os.path.join(REPO_ROOT, '02_data', 'label_sources')
        EMAIL = os.environ.get('NCBI_EMAIL', '')
        SLEEP_SEC = 0.35
        NCBI_RETMAX = 8
        OUTPUT_CSV = 'enhanced_sequence_dataset_balanced.csv'
        CHECKPOINT = '_checkpoint_balanced.csv'
        FEATURE_CACHE = '_new_gene_features_cache.csv'
        BASE_REPAIR_CACHE = '_base_repaired_cache.csv'
        ADD_NEGATIVE_INSTANCES = False
        ADD_USER_LABELED_GENES = True
        COMBINED_GENE_LABELS_FILE = os.path.join(LABEL_SOURCE_DIR, 'all_gene_labels_combined.csv')
        USER_GENE_LABELS_FILE = COMBINED_GENE_LABELS_FILE
        ADD_CGC_GENES = True
        CGC_FILE = os.path.join(RAW_SOURCE_DIR, 'Cosmic_CancerGeneCensus_Tsv_v103_GRCh37', 'Cosmic_CancerGeneCensus_v103_GRCh37.tsv.gz')
        CGC_TIER_FILTER = None
        ADD_EXTRA_POSITIVE_GENES = True
        EXTRA_POSITIVE_GENES_FILE = os.path.join(RAW_SOURCE_DIR, 'extra_positive_genes.txt')
        ADD_ONCOPLEX_GENE_LISTS = True
        ONCOPLEX_POSITIVE_FILE = os.path.join(RAW_SOURCE_DIR, 'ONCOPLEX-main', 'Data', 'Positive_pancancer.txt')
        ONCOPLEX_NEGATIVE_FILE = os.path.join(RAW_SOURCE_DIR, 'ONCOPLEX-main', 'Data', 'Negative_nondriver.csv')
        ADD_ONCOPLEX_NPG_GENES = True
        ONCOPLEX_NPG_DIR = os.path.join(RAW_SOURCE_DIR, 'ONCOPLEX-main', 'NPGs')
        ADD_ADDITIONAL_DRIVER_GENE_SETS = True
        ADDITIONAL_GENE_SETS_DIR = os.path.join(RAW_SOURCE_DIR, 'ONCOPLEX-main', 'Data', 'Additional gene sets')
        MAX_CANCERMINE_GENES = None
        MIN_CANCERMINE_CITATIONS = 3
        ADD_TCGA_BACKGROUND_INSTANCES = True
        MAX_TCGA_BACKGROUND_GENES = 300
        FETCH_ALL_AVAILABLE_POSITIVES = True
        FETCH_ALL_AVAILABLE_NEGATIVES = False
        TARGET_NEG_POS_RATIO = 1.4
        RECOMPUTE_EXISTING_NETWORK_ZEROS = True
        RECOMPUTE_EXISTING_METHYLATION_ZEROS = True
        OVERSAMPLE_POSITIVES_TO_BALANCE = False
        ADD_ADVANCED_DRIVER_FEATURES = False
        RANDOM_STATE = 42
        METHYLATION_COLS = ['Meth_Mean', 'Meth_STD', 'Meth_Hyper_Ratio', 'Meth_Hypo_Ratio', 'Prom_Meth_Mean', 'Prom_Meth_STD', 'Prom_Meth_Probe_Count']
        ADVANCED_FEATURE_PREFIX = 'Adv_'
        DROP_DERIVED_PROXY_FEATURES = True
        DROP_DERIVED_FEATURE_PATTERNS = ('Adv_', '_Proxy')
        DROP_ENGINEERED_SUFFIXES = ('_log', '_sqrt', '_sq', '_cb')
        DROP_ENGINEERED_EXACT = {'triple_net', 'ClinVar_net', 'ClinVar_ratio2', 'mut_diversity', 'GC_AT_ratio', 'CGC_PPI_ratio', 'PPI_path_ratio', 'meth_delta', 'prom_x_probe', 'meth_total', 'impact_score', 'trunc_impact', 'SNV_density', 'nonsense_ratio', 'frameshift_ratio', 'high_impact_SNV', 'ClinVar_burden', 'ClinVar_reviewed', 'CNV_delta', 'CNV_net'}
        DROP_FEATURE_PREFIXES = ('Adv_', 'IntOGen_', 'OncoPlex_Mutation_', 'OncoPlex_PanCancer_Mutation_')
        DROP_FEATURE_EXACT = {'IntOGen_Warning_Artifact_Count', 'IntOGen_Known_Artifact_Count', 'IntOGen_Domain_Hit_Count', 'IntOGen_2D_Cluster_Count', 'IntOGen_3D_Cluster_Count', 'IntOGen_Method_Count', 'IntOGen_SigMethod_Count'}
        INPUT_CSV = None
        for f in ['enhanced_sequence_dataset_multisource.csv', 'enhanced_sequence_dataset_extended.csv', 'enhanced_sequence_dataset_cleaned.csv']:
            if os.path.exists(f):
                INPUT_CSV = f
                break
        if INPUT_CSV is None:
            raise FileNotFoundError('No input CSV found!')
        MAF_FILE = os.path.join(RAW_SOURCE_DIR, 'mc3.v0.2.8.PUBLIC.maf.gz')
        ONCOPLEX_MUTATION_GENE_CANCER_FILE = os.path.join(RAW_SOURCE_DIR, 'ONCOPLEX-main', 'Data', 'mutation_genecancer.tsv')
        INTOGEN_DRIVERS_DIR = os.path.join(RAW_SOURCE_DIR, 'IntOGen-Drivers-20240920', '2024-06-18_IntOGen-Drivers')
        INTOGEN_COMPENDIUM_FILE = f'{INTOGEN_DRIVERS_DIR}/Compendium_Cancer_Genes.tsv'
        INTOGEN_UNFILTERED_FILE = f'{INTOGEN_DRIVERS_DIR}/Unfiltered_drivers.tsv'
        CNV_FILE = os.path.join(RAW_SOURCE_DIR, 'TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz')
        EXPR_FILE = os.path.join(RAW_SOURCE_DIR, 'EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz')
        METH_FILE = os.path.join(RAW_SOURCE_DIR, 'jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz')
        MANIFEST = os.path.join(RAW_SOURCE_DIR, 'humanmethylation450_15017482_v1-2.csv')
        CLINVAR_FILE = os.path.join(RAW_SOURCE_DIR, 'variant_summary.txt')
        PPI_LINKS = os.path.join(RAW_SOURCE_DIR, '9606.protein.links.full.v12.0.txt')
        PPI_ALIASES = os.path.join(RAW_SOURCE_DIR, '9606.protein.aliases.v12.0.txt')
        PATHWAY_FILE = os.path.join(RAW_SOURCE_DIR, 'Human_GOBP_AllPathways_noPFOCR_no_GO_iea_February_03_2026_symbol.txt')
        EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

        def E(p):
            return isinstance(p, str) and os.path.exists(p)

        def resolve_data_file(path):
            if not isinstance(path, str) or not path:
                return None
            if os.path.isfile(path):
                return path
            if os.path.isdir(path):
                nested = os.path.join(path, os.path.basename(path))
                if os.path.isfile(nested):
                    return nested
                files = [os.path.join(path, f) for f in os.listdir(path)]
                files = [f for f in files if os.path.isfile(f)]
                if files:
                    return files[0]
            gz_path = path + '.gz'
            if os.path.isfile(gz_path):
                return gz_path
            return None

        def safe_to_csv(df, path, **kwargs):
            try:
                df.to_csv(path, **kwargs)
                print(f'  Saved: {path}')
                return path
            except PermissionError:
                root, ext = os.path.splitext(path)
                alt = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
                df.to_csv(alt, **kwargs)
                print(f'  Permission denied for {path}')
                print(f'  File shayad Excel me open hai. Saved instead: {alt}')
                return alt

        def add_advanced_driver_features(df):
            out = df.copy()

            def s(col, default=0.0):
                if col in out.columns:
                    return pd.to_numeric(out[col], errors='coerce').fillna(default).astype(float)
                return pd.Series(default, index=out.index, dtype=float)

            def add(name, values):
                out[f'{ADVANCED_FEATURE_PREFIX}{name}'] = pd.Series(values, index=out.index).replace([np.inf, -np.inf], 0).fillna(0).astype(float)

            def z(series):
                series = pd.Series(series, index=out.index).astype(float)
                return (series - series.mean()) / (series.std(ddof=0) + 1e-06)
            ppi = s('PPI_degree')
            cgc = s('CGC_neighbor_count')
            pathway = s('pathway_count_all')
            snv = s('SNV_Count')
            missense = s('Missense')
            nonsense = s('Nonsense')
            frameshift = s('Frameshift')
            silent = s('Silent')
            splice = s('Splice_Site')
            patients = s('Patients_Affected')
            pathogenicity = s('Pathogenicity_Score')
            nonsilent_ratio = s('NonSilent_Ratio')
            trunc_ratio = s('Truncating_Ratio')
            impact = s('Mutation_Impact_Score')
            clin_path = s('ClinVar_Pathogenic_Count')
            clin_benign = s('ClinVar_Benign_Count')
            clin_total = s('ClinVar_Total_Variants')
            clin_reviewed = s('ClinVar_Reviewed_Pathogenic')
            clin_uncertain = s('ClinVar_Uncertain_Count')
            meth_mean = s('Meth_Mean')
            meth_std = s('Meth_STD')
            meth_hyper = s('Meth_Hyper_Ratio')
            meth_hypo = s('Meth_Hypo_Ratio')
            prom_mean = s('Prom_Meth_Mean')
            prom_std = s('Prom_Meth_STD')
            prom_probe = s('Prom_Meth_Probe_Count')
            cnv_gain = s('CNV_Gain_Ratio')
            cnv_loss = s('CNV_Loss_Ratio')
            cnv_amp = s('CNV_Amplification_Ratio')
            cnv_del = s('CNV_Deletion_Ratio')
            expr_mean = s('Expr_Mean')
            expr_cv = s('Expr_CV')
            gc_content = s('GC_Content')
            at_content = s('AT_Content')
            gc_skew = s('GC_Skew')
            at_skew = s('AT_Skew')
            cpg_density = s('CpG_Density')
            cpg_oe = s('CpG_OE')
            gc1 = s('GC1')
            gc2 = s('GC2')
            gc3 = s('GC3')
            entropy1 = s('Entropy_1mer')
            entropy2 = s('Entropy_2mer')
            seq_len = s('Seq_Length')
            protein_len = s('Protein_Length')
            hydro = s('Hydrophobic_Ratio')
            polar = s('Polar_Ratio')
            charged = s('Charged_Ratio')
            aromatic = s('Aromatic_Ratio')
            stop_frac = s('Stop_Codon_Fraction')
            hotspot_max = s('Hotspot_Max_Same_Position')
            hotspot_fraction = s('Hotspot_Fraction')
            protein_cluster = s('Protein_Position_Cluster_Score')
            domain_ratio = s('Domain_Mutation_Ratio')
            domain_unique = s('Domain_Unique_Count')
            deleterious_ratio = s('Deleterious_Impact_Ratio')
            mean_vaf = s('Mean_Tumor_VAF')
            context_entropy = s('Mutation_Context_Entropy')
            cpg_context = s('CpG_Context_Ratio')
            ct_context = s('C_to_T_Context_Ratio')
            oncoplex_max = s('OncoPlex_Mutation_MaxCancerRate')
            oncoplex_mean = s('OncoPlex_Mutation_MeanCancerRate')
            oncoplex_std = s('OncoPlex_Mutation_StdCancerRate')
            oncoplex_types = s('OncoPlex_Mutation_CancerTypes')
            oncoplex_burden = s('OncoPlex_PanCancer_Mutation_Burden')
            oncoplex_specificity = s('OncoPlex_Mutation_TissueSpecificity')
            intogen_q = s('IntOGen_Min_QValue')
            intogen_method_count = s('IntOGen_Method_Count')
            intogen_sig_method_count = s('IntOGen_SigMethod_Count')
            intogen_driver_cohorts = s('IntOGen_Driver_Cohort_Count')
            intogen_excess_mis = s('IntOGen_Max_Excess_Mis')
            intogen_excess_non = s('IntOGen_Max_Excess_Non')
            intogen_excess_spl = s('IntOGen_Max_Excess_Spl')
            intogen_domain_hits = s('IntOGen_Domain_Hit_Count')
            intogen_2d = s('IntOGen_2D_Cluster_Count')
            intogen_3d = s('IntOGen_3D_Cluster_Count')
            add('Network_Proximity_Index', np.log1p(ppi) * np.log1p(cgc + 1))
            add('CGC_Neighbor_Fraction', cgc / (ppi + 1))
            add('Pathway_Network_Load', np.log1p(pathway) * np.sqrt(ppi + 1))
            add('Cancer_Module_Bridge', (cgc + 1) * np.log1p(pathway) / np.sqrt(ppi + 1))
            add('ClinVar_Evidence_LogOdds', np.log1p(clin_path + clin_reviewed) - np.log1p(clin_benign + clin_uncertain))
            add('ClinVar_Pathogenic_Burden', (clin_path + 2 * clin_reviewed) / (clin_total + 1))
            add('ClinVar_Conflict_Index', clin_uncertain / (clin_total + 1))
            weighted_damage = missense + 2.5 * nonsense + 2.0 * frameshift
            add('Weighted_Mutation_Damage', weighted_damage)
            add('Patient_Normalized_Damage', weighted_damage / (patients + 1))
            add('Functional_Impact_Burden', np.log1p(snv) * (nonsilent_ratio + trunc_ratio) * np.log1p(pathogenicity + impact))
            add('Truncating_Enrichment', (nonsense + frameshift) / (snv + 1))
            add('Mutation_Recurrence_Index', np.log1p(snv) * np.log1p(patients + 1))
            add('Sample_Wise_Recurrence_Rate', patients / (snv + 1))
            add('Hotspot_Clustering_Index', np.log1p(hotspot_max) * (hotspot_fraction + protein_cluster))
            add('Functional_Impact_Severity', deleterious_ratio * np.log1p(weighted_damage + mean_vaf))
            add('Tumor_VAF_Weighted_Damage', mean_vaf * (1 + deleterious_ratio) * np.log1p(weighted_damage))
            add('Domain_Enrichment_Proxy', domain_ratio * np.log1p(domain_unique + snv))
            add('Selection_dNdS_Proxy', (missense + nonsense + frameshift + splice) / (silent + 1))
            add('Positive_Selection_Proxy', np.log1p(snv) * nonsilent_ratio * (1 + hotspot_fraction + domain_ratio))
            add('IntOGen_Selection_Evidence', -np.log10(intogen_q + 1e-300) * np.log1p(intogen_sig_method_count + intogen_driver_cohorts))
            add('IntOGen_Excess_Mutation_Load', intogen_excess_mis + 1.5 * intogen_excess_non + 1.2 * intogen_excess_spl)
            add('IntOGen_Cluster_Domain_Evidence', np.log1p(intogen_domain_hits + intogen_2d + intogen_3d))
            add('IntOGen_Method_Consensus', intogen_sig_method_count / (intogen_method_count + 1))
            add('Context_Background_Adjusted_Load', np.log1p(snv) / (1 + np.abs(gc_content - 0.5) + cpg_context))
            add('Trinucleotide_Context_Specificity', context_entropy * (1 + ct_context + cpg_context))
            add('Methylation_Dysregulation', np.abs(meth_hyper - meth_hypo) * np.log1p(prom_probe))
            add('Promoter_Methylation_Instability', prom_mean * (prom_std + meth_std) * np.log1p(prom_probe))
            add('Methylation_CNV_Crosstalk', np.abs(meth_hyper - meth_hypo) * (np.abs(cnv_gain - cnv_loss) + cnv_amp + cnv_del))
            add('CNV_Instability_Index', np.abs(cnv_gain - cnv_loss) + 2 * (cnv_amp + cnv_del))
            add('Expression_Variability_Load', np.log1p(np.maximum(expr_mean, 0)) * expr_cv)
            add('MultiOmic_Dysregulation_Load', z(weighted_damage) + z(cnv_amp + cnv_del) + z(expr_cv) + z(np.abs(meth_hyper - meth_hypo)))
            add('OncoPlex_PanCancer_Recurrence', np.log1p(oncoplex_burden) * np.log1p(oncoplex_types + 1))
            add('OncoPlex_Tissue_Specificity', oncoplex_specificity * np.log1p(oncoplex_max + 1))
            add('OncoPlex_Cancer_Spread', oncoplex_mean + oncoplex_std + np.log1p(oncoplex_types))
            add('GC_AT_Imbalance', np.abs(gc_content - at_content))
            add('Skew_Instability', np.abs(gc_skew) + np.abs(at_skew))
            add('CpG_Regulatory_Potential', cpg_density * (cpg_oe + 1) * (gc3 + 1))
            add('Sequence_Complexity_Index', entropy2 * (1 - np.minimum(np.abs(gc_content - 0.5) * 2, 1)))
            add('GC_Frame_Variance', pd.concat([gc1, gc2, gc3], axis=1).var(axis=1).fillna(0))
            add('GC3_Codon_Bias', gc3 - (gc1 + gc2) / 2)
            add('CpG_Suppression_Score', 1 / (cpg_oe + 1e-06))
            add('Entropy_Ratio_2mer_1mer', entropy2 / (entropy1 + 1e-06))
            add('Coding_Density_Proxy', protein_len * 3 / (seq_len + 1))
            add('Protein_Chemical_Complexity', hydro * polar + charged * aromatic)
            add('Stop_Codon_Penalty', stop_frac * np.log1p(seq_len))
            add('Sequence_Regulatory_Load', cpg_density * entropy2 * np.log1p(seq_len))
            add('Conservation_Structure_Proxy', (1 - stop_frac) * (entropy2 + domain_ratio + deleterious_ratio) * np.log1p(protein_len))
            add('Integrated_Driver_Pressure', np.log1p(ppi + cgc + pathway) * (1 + np.maximum(0, np.log1p(clin_path + clin_reviewed) - np.log1p(clin_benign + 1))) * (1 + nonsilent_ratio + trunc_ratio))
            return out

        def move_label_to_end(df):
            if 'Label' not in df.columns:
                return df
            cols = [c for c in df.columns if c != 'Label'] + ['Label']
            return df[cols]

        def drop_derived_proxy_features(df):
            if not DROP_DERIVED_PROXY_FEATURES:
                return df
            drop_cols = [c for c in df.columns if c not in ('Gene', 'Label') and (any((str(c).startswith(p) or str(c).endswith(p) for p in DROP_DERIVED_FEATURE_PATTERNS)) or any((str(c).startswith(p) for p in DROP_FEATURE_PREFIXES)) or str(c) in DROP_FEATURE_EXACT or (str(c) in DROP_ENGINEERED_EXACT) or str(c).endswith(DROP_ENGINEERED_SUFFIXES) or ('_x_' in str(c)) or str(c).startswith('IntOGen_Method_') or str(c).startswith('IntOGen_SigMethod_'))]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors='ignore')
                print(f'  Dropped derived/proxy features: {len(drop_cols)}')
            return df

        def merge_feature_frame(df, fdf):
            if fdf is None or fdf.empty or 'Gene' not in fdf.columns:
                return df
            out = df.copy()
            fdf = fdf.copy()
            out['Gene'] = out['Gene'].astype(str).str.upper()
            fdf['Gene'] = fdf['Gene'].astype(str).str.upper()
            add_cols = ['Gene'] + [c for c in fdf.columns if c != 'Gene' and c not in out.columns]
            if len(add_cols) <= 1:
                return out
            return out.merge(fdf[add_cols], on='Gene', how='left')

        def load_user_labeled_genes(path, existing_genes):
            if not ADD_USER_LABELED_GENES or not E(path):
                return []
            try:
                lab = pd.read_csv(path)
            except Exception as e:
                print(f'  User label file read error: {e}')
                return []
            if 'Gene' not in lab.columns or 'Label' not in lab.columns:
                print(f'  User label file ignored: needs Gene,Label columns ({path})')
                return []
            lab = lab[['Gene', 'Label']].dropna().copy()
            lab['Gene'] = lab['Gene'].astype(str).str.strip().str.upper()
            lab['Label'] = pd.to_numeric(lab['Label'], errors='coerce')
            lab = lab[lab['Label'].isin([0, 1])].copy()
            lab['Label'] = lab['Label'].astype(int)
            lab = lab.drop_duplicates('Gene', keep='first')
            lab = lab[~lab['Gene'].isin(existing_genes)]
            rows = list(lab[['Gene', 'Label']].itertuples(index=False, name=None))
            print(f"  User labeled genes available for fetch: {len(rows)} ({int((lab['Label'] == 1).sum())} pos, {int((lab['Label'] == 0).sum())} neg)")
            return rows

        def load_cgc_positive_genes(path, existing_genes):
            cgc_path = resolve_data_file(path)
            if not ADD_CGC_GENES or not cgc_path:
                return []
            try:
                cgc = pd.read_csv(cgc_path, sep='\t', compression='gzip', low_memory=False)
            except Exception as e:
                print(f'  CGC read error: {e}')
                return []
            gene_col = 'GENE_SYMBOL' if 'GENE_SYMBOL' in cgc.columns else None
            if gene_col is None:
                gene_col = next((c for c in cgc.columns if 'gene' in c.lower() and 'symbol' in c.lower()), None)
            if gene_col is None:
                print(f'  CGC ignored: gene symbol column not found in {cgc_path}')
                return []
            if CGC_TIER_FILTER is not None and 'TIER' in cgc.columns:
                cgc = cgc[pd.to_numeric(cgc['TIER'], errors='coerce') == int(CGC_TIER_FILTER)].copy()
            cgc['Gene'] = cgc[gene_col].astype(str).str.strip().str.upper()
            cgc = cgc[cgc['Gene'].ne('')].drop_duplicates('Gene')
            cgc = cgc[~cgc['Gene'].isin(existing_genes)]
            rows = [(g, 1) for g in cgc['Gene'].tolist()]
            tier_msg = 'all tiers' if CGC_TIER_FILTER is None else f'Tier {CGC_TIER_FILTER}'
            print(f'  CGC genes available for fetch: {len(rows)} positive genes ({tier_msg})')
            return rows

        def load_extra_positive_genes(path, existing_genes):
            if not ADD_EXTRA_POSITIVE_GENES or not E(path):
                return []
            try:
                if path.lower().endswith('.csv'):
                    raw = pd.read_csv(path)
                    if 'Gene' in raw.columns:
                        genes = raw['Gene'].astype(str).tolist()
                    else:
                        genes = raw.iloc[:, 0].astype(str).tolist()
                else:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    for ch in [',', ';', '\t', '\r', '\n']:
                        text = text.replace(ch, ' ')
                    genes = text.split()
            except Exception as e:
                print(f'  Extra positive gene file read error: {e}')
                return []
            bad_tokens = {'GENE', 'LABEL', '1', '0', 'NA', 'NAN'}
            cleaned = []
            for g in genes:
                g = str(g).strip().upper()
                if not g or g in bad_tokens:
                    continue
                if not any((c.isalpha() for c in g)):
                    continue
                cleaned.append(g)
            unique = list(dict.fromkeys(cleaned))
            rows = [(g, 1) for g in unique if g not in existing_genes]
            print(f'  Extra positive genes available for fetch: {len(rows)} from {path} ({len(unique)} unique in file)')
            return rows

        def _clean_gene_tokens(tokens):
            bad_tokens = {'GENE', 'LABEL', '1', '0', 'NA', 'NAN', 'NONE'}
            cleaned = []
            for g in tokens:
                g = str(g).strip().upper()
                if not g or g in bad_tokens:
                    continue
                if not any((c.isalpha() for c in g)):
                    continue
                cleaned.append(g)
            return list(dict.fromkeys(cleaned))

        def ncbi_header_matches_gene(header, gene):
            h = str(header).upper()
            g = str(gene).upper()
            if not h or not g:
                return False
            exact_symbol = re.escape(g)
            return bool(re.search(f'\\({exact_symbol}\\)', h) or re.search(f'\\[GENE={exact_symbol}\\]', h) or re.search(f'\\bGENE[:= ]{exact_symbol}\\b', h))

        def load_oncoplex_gene_list(path, label, existing_genes):
            if not ADD_ONCOPLEX_GENE_LISTS or not E(path):
                return []
            try:
                if path.lower().endswith('.csv'):
                    df = pd.read_csv(path)
                    gene_col = None
                    for c in df.columns:
                        vals = df[c].dropna().astype(str).head(20).tolist()
                        if vals and sum((any((ch.isalpha() for ch in v)) for v in vals)) >= max(1, len(vals) // 2):
                            gene_col = c
                            break
                    if gene_col is None:
                        gene_col = df.columns[-1]
                    genes = df[gene_col].astype(str).tolist()
                else:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    for ch in [',', ';', '\t', '\r', '\n']:
                        text = text.replace(ch, ' ')
                    genes = text.split()
            except Exception as e:
                print(f'  ONCOPLEX list read error ({path}): {e}')
                return []
            unique = _clean_gene_tokens(genes)
            rows = [(g, int(label)) for g in unique if g not in existing_genes]
            kind = 'positive' if int(label) == 1 else 'negative'
            print(f'  ONCOPLEX {kind} genes available for fetch: {len(rows)} from {path} ({len(unique)} unique in file)')
            return rows

        def load_oncoplex_npg_genes(folder, existing_genes):
            if not ADD_ONCOPLEX_NPG_GENES or not os.path.isdir(folder):
                return []
            genes = []
            file_counts = {}
            for fname in sorted(os.listdir(folder)):
                if not fname.upper().endswith('_NPG.TXT'):
                    continue
                path = os.path.join(folder, fname)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                except Exception:
                    continue
                for ch in [',', ';', '\t', '\r', '\n']:
                    text = text.replace(ch, ' ')
                vals = _clean_gene_tokens(text.split())
                file_counts[fname] = len(vals)
                genes.extend(vals)
            unique = list(dict.fromkeys(genes))
            rows = [(g, 1) for g in unique if g not in existing_genes]
            print(f'  ONCOPLEX NPG genes available for fetch: {len(rows)} from {len(file_counts)} cancer-type files ({len(unique)} unique)')
            return rows

        def load_additional_driver_gene_sets(folder, existing_genes):
            if not ADD_ADDITIONAL_DRIVER_GENE_SETS or not os.path.isdir(folder):
                return []
            rows = []
            oncokb_path = os.path.join(folder, 'oncoKB_genes.tsv')
            if E(oncokb_path):
                try:
                    ok = pd.read_csv(oncokb_path, sep='\t', low_memory=False)
                    col = 'Hugo Symbol' if 'Hugo Symbol' in ok.columns else ok.columns[0]
                    genes = _clean_gene_tokens(ok[col].dropna().astype(str).tolist())
                    ok_rows = [(g, 1) for g in genes if g not in existing_genes]
                    rows.extend(ok_rows)
                    print(f'  Additional OncoKB genes available: {len(ok_rows)} ({len(genes)} unique)')
                except Exception as e:
                    print(f'  OncoKB additional set skipped: {e}')
            ongene_path = os.path.join(folder, 'ongene_human.txt')
            if E(ongene_path):
                try:
                    og = pd.read_csv(ongene_path, sep='\t', low_memory=False)
                    col = 'OncogeneName' if 'OncogeneName' in og.columns else og.columns[1]
                    genes = _clean_gene_tokens(og[col].dropna().astype(str).tolist())
                    og_rows = [(g, 1) for g in genes if g not in existing_genes]
                    rows.extend(og_rows)
                    print(f'  Additional Ongene genes available: {len(og_rows)} ({len(genes)} unique)')
                except Exception as e:
                    print(f'  Ongene additional set skipped: {e}')
            cm_path = os.path.join(folder, 'cancermine_collated.tsv')
            if E(cm_path):
                try:
                    cm = pd.read_csv(cm_path, sep='\t', low_memory=False)
                    cm['gene_normalized'] = cm['gene_normalized'].astype(str).str.upper()
                    cm['citation_count'] = pd.to_numeric(cm['citation_count'], errors='coerce').fillna(0)
                    cm = cm[cm['role'].isin(['Oncogene', 'Tumor_Suppressor', 'Driver'])].copy()
                    cm = cm[cm['citation_count'] >= MIN_CANCERMINE_CITATIONS]
                    gene_scores = cm.groupby('gene_normalized', as_index=False)['citation_count'].max().sort_values('citation_count', ascending=False)
                    genes = _clean_gene_tokens(gene_scores['gene_normalized'].tolist())
                    cm_rows = [(g, 1) for g in genes if g not in existing_genes]
                    if MAX_CANCERMINE_GENES is not None:
                        cm_rows = cm_rows[:MAX_CANCERMINE_GENES]
                    rows.extend(cm_rows)
                    print(f'  Additional CancerMine genes available: {len(cm_rows)} (cap={MAX_CANCERMINE_GENES}, min citations={MIN_CANCERMINE_CITATIONS})')
                except Exception as e:
                    print(f'  CancerMine additional set skipped: {e}')
            rows = list(dict.fromkeys(rows))
            print(f'  Additional driver gene sets total available: {len(rows)}')
            return rows

        def get_tcga_background_gene_labels(existing_genes, known_positive_genes, already_planned):
            if not ADD_TCGA_BACKGROUND_INSTANCES or not E(MAF_FILE):
                return []
            print('\n  Getting extra TCGA background genes...')
            try:
                m = pd.read_csv(MAF_FILE, sep='\t', compression='gzip', low_memory=False, usecols=['Hugo_Symbol', 'Tumor_Sample_Barcode', 'FILTER'])
            except Exception as e:
                print(f'  TCGA background skipped: {e}')
                return []
            m = m[m['FILTER'] == 'PASS'].copy()
            m['Hugo_Symbol'] = m['Hugo_Symbol'].astype(str).str.upper()
            gc_m = m.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique()
            bg = gc_m[(gc_m >= 5) & (gc_m <= 80) & ~gc_m.index.isin(existing_genes) & ~gc_m.index.isin(known_positive_genes) & ~gc_m.index.isin(already_planned)].sort_values()
            if bg.empty:
                print('  TCGA background: 0 genes')
                return []
            mid = bg.iloc[len(bg) // 5:4 * len(bg) // 5]
            n = len(mid) if MAX_TCGA_BACKGROUND_GENES is None else min(MAX_TCGA_BACKGROUND_GENES, len(mid))
            bg_sel = mid.sample(n=n, random_state=RANDOM_STATE) if n > 0 else mid.iloc[:0]
            rows = [(str(g).upper(), 0) for g in bg_sel.index]
            print(f'  TCGA background: {len(rows)} real Label=0 genes')
            return rows
        ALL_NEW_POSITIVE = ['ABL1', 'ACVR1', 'AGO2', 'AKT1', 'AKT2', 'AKT3', 'ALK', 'ALOX12B', 'ANKRD11', 'APC', 'AR', 'ARAF', 'ARFRP1', 'ARID1A', 'ARID1B', 'ARID2', 'ARID5B', 'ASXL1', 'ASXL2', 'ATM', 'ATR', 'ATRX', 'AURKA', 'AURKB', 'AXIN1', 'AXIN2', 'AXL', 'B2M', 'BAP1', 'BARD1', 'BCL10', 'BCL2', 'BCL2L1', 'BCL2L11', 'BCL6', 'BCOR', 'BCORL1', 'BLM', 'BMPR1A', 'BRAF', 'BRCA1', 'BRCA2', 'BRD4', 'BRIP1', 'BTG1', 'BTK', 'CALR', 'CARD11', 'CASP8', 'CBFB', 'CBL', 'CBLB', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CD274', 'CD79A', 'CD79B', 'CDC73', 'CDH1', 'CDK12', 'CDK4', 'CDK6', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B', 'CDKN2C', 'CEBPA', 'CHEK1', 'CHEK2', 'CIC', 'CREBBP', 'CRKL', 'CRLF2', 'CSF1R', 'CSF3R', 'CTCF', 'CTNNB1', 'CUL3', 'CYLD', 'DAXX', 'DDR2', 'DICER1', 'DIS3', 'DNMT1', 'DNMT3A', 'DNMT3B', 'DOT1L', 'EED', 'EGFR', 'EIF1AX', 'EP300', 'ERBB2', 'ERBB3', 'ERBB4', 'ESR1', 'ETV1', 'ETV6', 'EZH1', 'EZH2', 'FANCA', 'FANCC', 'FANCD2', 'FANCG', 'FANCI', 'FANCL', 'FAS', 'FBXW7', 'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'FH', 'FLCN', 'FLT1', 'FLT3', 'FLT4', 'FOXA1', 'FOXL2', 'FOXO1', 'FOXP1', 'GATA1', 'GATA2', 'GATA3', 'GNA11', 'GNAQ', 'GNAS', 'H3F3A', 'H3F3B', 'HDAC1', 'HDAC2', 'HDAC3', 'HGF', 'HIST1H1C', 'HIST1H2BD', 'HIST1H3B', 'HNF1A', 'HNF1B', 'HRAS', 'IDH1', 'IDH2', 'IKZF1', 'IL6ST', 'IL7R', 'IRF4', 'JAK1', 'JAK2', 'JAK3', 'KDM5A', 'KDM5C', 'KDM6A', 'KDR', 'KEAP1', 'KIT', 'KMT2A', 'KMT2B', 'KMT2C', 'KMT2D', 'KMT2E', 'KRAS', 'LATS1', 'LATS2', 'LMO1', 'MAF', 'MAFB', 'MAP2K1', 'MAP2K2', 'MAP2K4', 'MAP3K1', 'MAPK1', 'MCL1', 'MDM2', 'MDM4', 'MED12', 'MEF2B', 'MEN1', 'MET', 'MITF', 'MLH1', 'MPL', 'MRE11A', 'MSH2', 'MSH3', 'MSH6', 'MTOR', 'MUTYH', 'MYC', 'MYCL', 'MYCN', 'MYD88', 'NBN', 'NCOA3', 'NCOR1', 'NF1', 'NF2', 'NFE2L2', 'NOTCH1', 'NOTCH2', 'NOTCH3', 'NOTCH4', 'NPM1', 'NRAS', 'NSD1', 'PALB2', 'PAX5', 'PBRM1', 'PDGFRA', 'PDGFRB', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PIM1', 'PMS2', 'POLD1', 'POLE', 'PPARG', 'PPP2R1A', 'PRDM1', 'PRKAR1A', 'PTCH1', 'PTEN', 'PTPN11', 'RAC1', 'RAD21', 'RAD51', 'RAD51B', 'RAD51C', 'RAD51D', 'RAF1', 'RARA', 'RASA1', 'RB1', 'RBM10', 'RET', 'RHOA', 'RICTOR', 'RIT1', 'RNF43', 'ROS1', 'RUNX1', 'RUNX1T1', 'RXRA', 'SF3B1', 'SMAD2', 'SMAD3', 'SMAD4', 'SMARCA4', 'SMARCB1', 'SMO', 'SOCS1', 'SOX2', 'SPOP', 'SRSF2', 'STAG2', 'STAT3', 'STAT5A', 'STAT5B', 'STK11', 'SUFU', 'SYK', 'TAL1', 'TCF7L2', 'TERT', 'TET2', 'TGFBR1', 'TGFBR2', 'TNFAIP3', 'TP53', 'TSC1', 'TSC2', 'U2AF1', 'VHL', 'WRN', 'WT1', 'YAP1', 'ZRSR2', 'VEGFA', 'VEGFB', 'VEGFC', 'FLT1', 'KDR', 'HIF1A', 'EPAS1', 'BCL2L1', 'BIRC5', 'CDKN1A', 'CDKN1B', 'CDKN1C', 'CDKN2C', 'CDKN2D', 'RBL1', 'RBL2', 'RB1CC1', 'E2F1', 'E2F2', 'E2F3', 'E2F4', 'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7', 'CDK1', 'CDK2', 'CDK7', 'CCNA2', 'CCNB1', 'CCNE2', 'MRE11A', 'RAD50', 'NBN', 'FAT4', 'FAT3', 'FAT1', 'RHOA', 'RAC1', 'GNA11', 'CIC', 'FUBP1', 'ARID2', 'KDM5A', 'SETDB1', 'MAP2K1', 'AKT2', 'AKT3', 'PIK3R2', 'SMAD2', 'SMAD3', 'PTPRD', 'LATS1', 'LATS2', 'YAP1', 'WWTR1', 'HNF1A', 'FOXA2', 'GATA4', 'GATA6', 'CARD11', 'CD79A', 'MYD88', 'PAX5', 'IRF4', 'BCL6', 'MAF', 'MAFB', 'CCND2', 'CCND3', 'ACVR1', 'BMPR1A', 'SDHA', 'SDHB', 'SDHC', 'SDHD', 'FH', 'MDH2', 'ACO2', 'DROSHA', 'DGCR8', 'SPOP', 'CUL3', 'ERBB3', 'ERBB4', 'FGFR4', 'IGF1R', 'TCF7L2', 'SOX9', 'AJUBA', 'GNA13', 'MGA', 'EPHA2', 'SYNE1', 'MAP3K13', 'TBX3', 'RBM10', 'ACVR2A', 'RXRA', 'TSC1', 'CDKN1A', 'BIRC3', 'BMPR2', 'BRIP1', 'CD274', 'CDK6', 'CHD1', 'EML4', 'EPHB3', 'EPHB4', 'FUS', 'HGF', 'HMGA2', 'KIF1B', 'MAP2K2', 'HOXA5', 'PRCC', 'SS18', 'EWSR1', 'PAX3', 'PAX7', 'CBFB', 'PML', 'RARA', 'ACVR1B', 'AFF1', 'AFF3', 'ATF1', 'ATIC', 'BCOR', 'BCL3', 'BCL9', 'BCL11A', 'BCL11B', 'BTG1', 'CARS', 'CDX2', 'CEBPB', 'CEBPD', 'CHN1', 'CNOT3', 'CUX1', 'DDIT3', 'DDX6', 'DEK', 'EBF1', 'EIF3E', 'ELF3', 'ELK4', 'EP400', 'ERC1', 'ETV4', 'ETV5', 'EXT1', 'EXT2', 'FCRL4', 'FIP1L1', 'FLI1', 'FOXO3', 'FOXO4', 'FOXP1', 'GID4', 'GNA12', 'GNA13', 'GPC3', 'HMGA1', 'HOOK3', 'HOXA9', 'IL6ST', 'IL7R', 'JAK3', 'KAT6A', 'KAT6B', 'KLF2', 'LASP1', 'LCK', 'LEF1', 'LMO1', 'LMO2', 'MALAT1', 'MAML2', 'MAP2K4', 'MEF2C', 'METTL14', 'MLLT1', 'MLLT3', 'MLLT4', 'MN1', 'MSI2', 'MUC1', 'MYH11', 'NFIB', 'NSD1', 'NSD2', 'NTRK1', 'NTRK3', 'NUMA1', 'NUP214', 'NUP98', 'NUTM1', 'P2RY8', 'PAX8', 'PCBP1', 'PDE4DIP', 'PICALM', 'PIK3CG', 'PLCG1', 'POT1', 'PPARG', 'PPP2R2A', 'PRDM16', 'PRDM2', 'PRKAR1A', 'PSIP1', 'PTPRC', 'QKI', 'RAC2', 'RALGDS', 'RBM15', 'RRM2B', 'SBDS', 'SETBP1', 'SFPQ', 'SH2B3', 'SLC34A2', 'SMARCA2', 'SPEN', 'SS18L1', 'SSX1', 'SSX2', 'STAT6', 'STIL', 'SYK', 'TAL1', 'TAL2', 'TCF12', 'TCF3', 'TCF4', 'TFEB', 'THRA', 'TNFRSF14', 'TOP1', 'TRIM24', 'TRIM27', 'TRIM33', 'USP6', 'VAV1', 'WAS', 'WNT5A', 'XPO1', 'YWHAE', 'ZBTB16', 'ZMYM2', 'ZNF521', 'ZRSR2', 'NRG1', 'NRG2', 'PDGFA', 'PDGFB', 'IGF1', 'IGF2', 'FGF1', 'FGF2', 'FGF7', 'CDH2', 'TWIST1', 'TWIST2', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2', 'MMP2', 'MMP9', 'ANGPT1', 'TEK', 'TERT', 'DKC1', 'PARP1', 'PARP2', 'XRCC1', 'LIG4', 'XRCC4', 'B2M', 'IDO1', 'CIITA', 'HK2', 'LDHA', 'G6PD', 'PFKFB3', 'GOT1', 'PHGDH', 'FAP', 'CXCL12', 'CXCR4', 'ACVR2B', 'MEN2', 'RET', 'FLCN', 'SLX4', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'XPA', 'XPC', 'NTHL1', 'MSH3', 'EPCAM', 'LZTR1', 'DDX3X', 'SPEN', 'NCOR1', 'ZBTB7A', 'TNFRSF14', 'IKBKB', 'TNFAIP3', 'BIRC6', 'GNA12', 'RHOH', 'MYH9', 'GPS2', 'MYST3', 'NOTCH3', 'MED23', 'SETD2', 'ARID1A', 'SMARCA4', 'U2AF1', 'RIT1', 'CBFB', 'TBX3', 'GATA3', 'MAP3K1', 'NCOR1', 'RUNX1', 'ERBB2']
        NCG_CANDIDATES_NEG = ['HSP90AA1', 'HSP90AB1', 'HSPA1A', 'HSPA4', 'HSPA5', 'HSPA8', 'HSPA9', 'HSPD1', 'HDAC4', 'HDAC5', 'HDAC6', 'HDAC7', 'HDAC8', 'HDAC9', 'HDAC10', 'SIRT1', 'SIRT2', 'SIRT3', 'SIRT4', 'SIRT5', 'SIRT6', 'SIRT7', 'CDK3', 'CDK5', 'CDK10', 'CDK11A', 'CCNA1', 'CCNT1', 'CCNT2', 'E2F6', 'E2F7', 'E2F8', 'TOP2B', 'BUB1', 'BUB1B', 'BUB3', 'MAD1L1', 'MAD2L1', 'PLK2', 'PLK3', 'PLK4', 'MAP3K5', 'MAP3K7', 'MAP3K8', 'MAP4K1', 'MAP4K4', 'PDPK1', 'RPS6KA1', 'RPS6KA2', 'RPS6KA5', 'RPS6KB1', 'SGK2', 'SGK3', 'MLST8', 'PRKCA', 'PRKCE', 'PRKCG', 'PRKCH', 'PRKCI', 'PRKCQ', 'PRKCZ', 'PRKDC', 'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'PRKAG3', 'BLK', 'HCK', 'FGR', 'FRK', 'ARG', 'FES', 'FER', 'TYK2', 'STAT1', 'STAT2', 'STAT4', 'STAT6', 'SMAD1', 'SMAD5', 'TGFBR3', 'BMPR2', 'WNT1', 'WNT2', 'WNT3', 'WNT5B', 'WNT6', 'WNT7A', 'WNT7B', 'WNT8A', 'WNT9A', 'WNT9B', 'WNT10A', 'WNT10B', 'WNT11', 'WNT16', 'FZD1', 'FZD2', 'FZD4', 'FZD5', 'FZD6', 'FZD7', 'FZD8', 'FZD9', 'DVL1', 'DVL2', 'DVL3', 'GSK3A', 'DLL1', 'DLL3', 'JAG1', 'JAG2', 'HEY2', 'HES1', 'HES5', 'MAML1', 'MAML3', 'EGF', 'EREG', 'AREG', 'TGFA', 'HBEGF', 'BTC', 'FGF9', 'FGF13', 'IGF2R', 'PDGFC', 'PDGFD', 'USP7', 'HIPK2', 'CDKN2D', 'RB1CC1', 'ABRAXAS1', 'ATRIP', 'TOPBP1', 'MLH3', 'MSH4', 'MSH5', 'EPHA1', 'EPHA4', 'EPHA5', 'EPHB3', 'FOS', 'FOSB', 'JUN', 'JUNB', 'JUND', 'NFKB1', 'NFKB2', 'RELA', 'RELB', 'BCL2A1', 'BID', 'BAD', 'BAK1', 'CFLAR', 'CASP1', 'CASP2', 'CASP6', 'CASP7', 'PARP1', 'TNKS']
        TUMOR_SUPPRESSORS_NEG = ['CABLES2', 'CCNG1', 'CCNG2', 'GADD45A', 'GADD45B', 'GADD45G', 'XRCC1', 'XRCC2', 'XRCC3', 'XRCC5', 'XRCC6', 'LIG1', 'LIG3', 'LIG4', 'PNKP', 'APTX', 'APLF', 'POLL', 'POLB', 'PMAIP1', 'BBC3', 'BIK', 'BNIP3', 'BNIP3L', 'HRK', 'BOK', 'BMF', 'BAD', 'DAPK1', 'DAPK2', 'DAPK3', 'RASSF1', 'RASSF2', 'RASSF4', 'RASSF5', 'APC2', 'AXIN2', 'CSNK1A1', 'DKK1', 'DKK3', 'DKK4', 'SFRP1', 'SFRP2', 'SFRP3', 'SFRP5', 'CTNNBIP1', 'PHLPP1', 'PHLPP2', 'INPP5D', 'SYNJ1', 'NPRL2', 'NPRL3', 'SPRED1', 'SPRED2', 'SPRED3', 'RASA3', 'DAB2IP', 'GAB1', 'GAB2', 'RUNX3', 'HOXA5', 'HOXD10', 'WWOX', 'PRDM5', 'IRF3', 'IRF5', 'IRF7', 'IRF8', 'IRF9', 'NDRG2', 'NDRG3', 'NDRG4', 'CDH13', 'CDH15', 'CLDN3', 'CLDN4', 'CLDN7', 'OCLN', 'TJP1']
        HOUSEKEEPING_NEG = ['GAPDH', 'LDHA', 'LDHB', 'PGK1', 'PGK2', 'PGM1', 'ALDOC', 'TPI1', 'PGAM1', 'PGAM2', 'ENO2', 'ENO3', 'PKM', 'PKLR', 'HK1', 'HK2', 'GPI', 'PFKL', 'PFKM', 'PFKP', 'CS', 'ACO1', 'IDH3A', 'IDH3B', 'SUCLA2', 'SUCLG1', 'UQCRB', 'UQCRC1', 'CYCS', 'COX4I1', 'COX5A', 'COX5B', 'COX6A1', 'COX6B1', 'RPL3', 'RPL4', 'RPL6', 'RPL7', 'RPL8', 'RPL9', 'RPL11', 'RPL12', 'RPL13', 'RPS3', 'RPS4X', 'RPS5', 'RPS6', 'RPS7', 'RPS8', 'RPS9', 'RPS11', 'RPS12', 'EIF2A', 'EIF2B1', 'EIF3B', 'EIF3C', 'EIF4B', 'PSMA1', 'PSMA2', 'PSMA3', 'PSMA4', 'PSMA5', 'PSMA6', 'PSMA7', 'PSMB1', 'PSMB2', 'PSMB3', 'PSMB4', 'PSMB5', 'PSMB6', 'PSMB7', 'PSMC1', 'PSMC2', 'PSMC3', 'PSMC4', 'PSMC5', 'PSMC6', 'PSMD1', 'PSMD2', 'PSMD3', 'PSMD4', 'PSMD6', 'PSMD7', 'PSMD11', 'PSMD12', 'ACTB', 'ACTG2', 'ACTA1', 'ACTA2', 'ACTN1', 'ACTN3', 'ACTN4', 'TUBB', 'TUBB2A', 'TUBB2B', 'TUBB3', 'TUBB4A', 'TUBB4B', 'TUBB6', 'TUBA1A', 'TUBA1B', 'TUBA1C', 'TUBA4A', 'VIM', 'DES', 'GFAP', 'NES']
        CODON_TABLE = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T', 'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R', 'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P', 'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A', 'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G', 'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L', 'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*', 'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'}
        HYDRO = set('AILMFWV')
        POLAR = set('STNQY')
        CHARGED = set('KRDE')
        AROMATIC = set('FWY')

        def translate(seq):
            aa = ''
            for i in range(0, len(seq) - 2, 3):
                aa += CODON_TABLE.get(seq[i:i + 3], 'X')
            return aa

        def gc_at_pos(seq, pos):
            v = [seq[i] for i in range(pos, len(seq), 3)]
            return sum((1 for x in v if x in 'GC')) / max(len(v), 1)

        def cpg_oe(seq):
            c, g, cg = (seq.count('C'), seq.count('G'), seq.count('CG'))
            exp = c * g / max(len(seq), 1)
            return cg / exp if exp > 0 else 0

        def seq_features(gene, seq):
            seq = seq.upper().replace('\n', '').replace('\r', '')
            L = max(len(seq), 1)
            aa = translate(seq)
            freq = Counter(seq)
            G, C, A, T = (seq.count('G'), seq.count('C'), seq.count('A'), seq.count('T'))
            ent1 = -sum((v / L * log2(v / L) for v in freq.values() if v > 0))
            dc = Counter((seq[i:i + 2] for i in range(L - 1)))
            tot2 = max(L - 1, 1)
            ent2 = -sum((v / tot2 * log2(v / tot2) for v in dc.values() if v > 0))
            row = {'Gene': gene, 'Seq_Length': L, 'GC_Content': (G + C) / L, 'AT_Content': (A + T) / L, 'GC_Skew': (G - C) / max(1, G + C), 'AT_Skew': (A - T) / max(1, A + T), 'Entropy_1mer': ent1, 'Entropy_2mer': ent2, 'CpG_Density': seq.count('CG') / L, 'Protein_Length': len(aa), 'Hydrophobic_Ratio': sum((a in HYDRO for a in aa)) / max(1, len(aa)), 'Polar_Ratio': sum((a in POLAR for a in aa)) / max(1, len(aa)), 'Charged_Ratio': sum((a in CHARGED for a in aa)) / max(1, len(aa)), 'Aromatic_Ratio': sum((a in AROMATIC for a in aa)) / max(1, len(aa)), 'Stop_Codon_Fraction': aa.count('*') / max(1, len(aa)), 'Seq_Length_new': L, 'GC_Content_new': (G + C) / L, 'AT_Content_new': (A + T) / L, 'GC1': gc_at_pos(seq, 0), 'GC2': gc_at_pos(seq, 1), 'GC3': gc_at_pos(seq, 2), 'CpG_OE': cpg_oe(seq)}
            aa_len = max(1, len(aa))
            codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
            valid_codons = [c for c in codons if len(c) == 3 and set(c) <= set('ACGT')]
            codon_total = max(1, len(valid_codons))
            aa_counts = Counter(aa)
            row.update({'Seq_N_Content': sum((1 for x in seq if x not in 'ACGT')) / L, 'Seq_Purine_Ratio': (A + G) / L, 'Seq_Pyrimidine_Ratio': (C + T) / L, 'Seq_GC_AT_Odds': (G + C + 1) / (A + T + 1), 'Seq_CpG_Count': seq.count('CG'), 'Seq_CpG_per_kb': seq.count('CG') / (L / 1000), 'Seq_Longest_ORF_Proxy': max((len(x) for x in aa.split('*')), default=0), 'Seq_Internal_Stop_Count': max(aa.count('*') - 1, 0), 'AA_Aliphatic_Ratio': sum((a in set('AILV') for a in aa)) / aa_len, 'AA_Acidic_Ratio': sum((a in set('DE') for a in aa)) / aa_len, 'AA_Basic_Ratio': sum((a in set('KRH') for a in aa)) / aa_len, 'AA_Proline_Ratio': aa_counts.get('P', 0) / aa_len, 'AA_Glycine_Ratio': aa_counts.get('G', 0) / aa_len, 'AA_Cysteine_Ratio': aa_counts.get('C', 0) / aa_len, 'AA_Instability_Proxy': (aa_counts.get('P', 0) + aa_counts.get('G', 0) + aa_counts.get('C', 0)) / aa_len})
            codon_groups = {'Codon_Stop_Ratio': ['TAA', 'TAG', 'TGA'], 'Codon_Start_ATG_Ratio': ['ATG'], 'Codon_CpG_Rich_Ratio': ['CGT', 'CGC', 'CGA', 'CGG', 'ACG', 'TCG', 'GCG', 'CCG'], 'Codon_GC_Ending_Ratio': [c for c in CODON_TABLE if c[-1] in 'GC'], 'Codon_AT_Ending_Ratio': [c for c in CODON_TABLE if c[-1] in 'AT'], 'Codon_Rare_Arg_Proxy': ['CGA', 'CGG', 'CGT', 'CGC', 'AGA', 'AGG']}
            codon_counts = Counter(valid_codons)
            for name, cods in codon_groups.items():
                row[name] = sum((codon_counts.get(c, 0) for c in cods)) / codon_total
            di_tot = max(L - 1, 1)
            for k, v in Counter((seq[i:i + 2] for i in range(L - 1))).items():
                row[f'di_{k}'] = v / di_tot
            k3_tot = max(L - 2, 1)
            for k, v in Counter((seq[i:i + 3] for i in range(L - 2))).items():
                row[f'k3_{k}'] = v / k3_tot
            return row

        def fetch_ncbi(gene, email=EMAIL, sleep=SLEEP_SEC):
            terms = [f'{gene}[Gene] AND "Homo sapiens"[Organism] AND srcdb_refseq[PROP] AND biomol_mrna[PROP]', f'{gene}[Gene] AND "Homo sapiens"[Organism] AND (mRNA[Filter] OR RefSeq[Filter])', f'{gene}[Gene] AND "Homo sapiens"[Organism] AND biomol_mrna[PROP]', f'{gene}[Gene] AND "Homo sapiens"[Organism] AND nuccore_refseq[PROP]']
            seen = set()
            for term in terms:
                p = {'db': 'nuccore', 'term': term, 'retmax': str(NCBI_RETMAX), 'retmode': 'xml', 'email': email}
                try:
                    xml = urllib.request.urlopen(f'{EUTILS}/esearch.fcgi?' + urllib.parse.urlencode(p), timeout=30).read().decode()
                    time.sleep(sleep)
                    ids = [e.text for e in ET.fromstring(xml).findall('.//IdList/Id') if e.text]
                except Exception:
                    continue
                ids = [x for x in ids if x not in seen]
                seen.update(ids)
                if not ids:
                    continue
                best = None
                for rid in ids:
                    p2 = {'db': 'nuccore', 'id': rid, 'rettype': 'fasta', 'retmode': 'text', 'email': email}
                    try:
                        fasta = urllib.request.urlopen(f'{EUTILS}/efetch.fcgi?' + urllib.parse.urlencode(p2), timeout=30).read().decode().strip()
                        time.sleep(sleep)
                    except Exception:
                        continue
                    if not fasta.startswith('>'):
                        continue
                    lines = fasta.splitlines()
                    hdr = lines[0][1:].strip()
                    seq = ''.join((l.strip() for l in lines[1:] if l and (not l.startswith('>')))).upper()
                    seq = ''.join((ch for ch in seq if ch.isalpha()))
                    if len(seq) < 150:
                        continue
                    if not ncbi_header_matches_gene(hdr, gene):
                        continue
                    score = len(seq)
                    h = hdr.upper()
                    if 'NM_' in h:
                        score += 100000
                    if 'MANE' in h or 'REFSEQ' in h:
                        score += 20000
                    if 'PREDICTED' in h or 'LOW QUALITY' in h:
                        score -= 50000
                    if best is None or score > best[0]:
                        best = (score, hdr, seq)
                if best is not None:
                    return (best[1], best[2])
            return (None, None)

        def get_mutation_features(gene_list):
            if not E(MAF_FILE):
                return pd.DataFrame()
            print('  Loading mutations...')
            maf = pd.read_csv(MAF_FILE, sep='\t', compression='gzip', low_memory=False, usecols=['Hugo_Symbol', 'Variant_Classification', 'Tumor_Sample_Barcode', 'FILTER'])
            maf = maf[(maf['FILTER'] == 'PASS') & maf['Hugo_Symbol'].isin(set(gene_list))]

            def s(x):
                t = len(x)
                vc = x['Variant_Classification']
                ns = vc.isin(['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']).sum()
                tr = vc.isin(['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins']).sum()
                return pd.Series({'SNV_Count': t, 'Missense': (vc == 'Missense_Mutation').sum(), 'Nonsense': (vc == 'Nonsense_Mutation').sum(), 'Frameshift': vc.str.contains('Frame', na=False).sum(), 'Silent': (vc == 'Silent').sum(), 'Splice_Site': (vc == 'Splice_Site').sum(), 'Patients_Affected': x['Tumor_Sample_Barcode'].nunique(), 'NonSilent_Ratio': ns / max(1, t), 'Truncating_Ratio': tr / max(1, t), 'Pathogenicity_Score': tr + ns, 'High_Impact_Ratio': tr / max(1, t), 'Missense_Ratio': (vc == 'Missense_Mutation').sum() / max(1, t), 'Silent_Ratio': (vc == 'Silent').sum() / max(1, t), 'Mutation_Impact_Score': 2 * tr + ns})
            df = maf.groupby('Hugo_Symbol').apply(s).reset_index()
            df.rename(columns={'Hugo_Symbol': 'Gene'}, inplace=True)
            print(f'    Mutation: {len(df)} genes')
            return df

        def get_mutation_context_domain_features(gene_list):
            if not E(MAF_FILE):
                return pd.DataFrame()
            print('  Loading advanced mutation context/domain features...')
            usecols = ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'FILTER', 'Variant_Classification', 'Variant_Type', 'Start_Position', 'Protein_position', 'SIFT', 'PolyPhen', 'DOMAINS', 'CONTEXT', 't_depth', 't_alt_count']
            try:
                maf = pd.read_csv(MAF_FILE, sep='\t', compression='gzip', low_memory=False, usecols=usecols)
            except ValueError:
                maf = pd.read_csv(MAF_FILE, sep='\t', compression='gzip', low_memory=False)
                maf = maf[[c for c in usecols if c in maf.columns]]
            if maf.empty or 'Hugo_Symbol' not in maf.columns:
                return pd.DataFrame()
            genes = {str(g).upper() for g in gene_list}
            maf['Hugo_Symbol'] = maf['Hugo_Symbol'].astype(str).str.upper()
            if 'FILTER' in maf.columns:
                maf = maf[maf['FILTER'].astype(str).eq('PASS')]
            maf = maf[maf['Hugo_Symbol'].isin(genes)].copy()
            if maf.empty:
                print('    Advanced mutation: 0 genes')
                return pd.DataFrame()

            def entropy(vals):
                counts = pd.Series(vals).dropna().astype(str).value_counts()
                if counts.empty:
                    return 0.0
                p = counts / counts.sum()
                return float(-(p * np.log2(p)).sum())

            def protein_pos(v):
                text = str(v)
                nums = re.findall('\\d+', text)
                return float(nums[0]) if nums else np.nan
            maf['Protein_Pos_Num'] = maf.get('Protein_position', pd.Series(index=maf.index, dtype=object)).map(protein_pos)
            maf['Start_Position'] = pd.to_numeric(maf.get('Start_Position', 0), errors='coerce')
            maf['t_depth'] = pd.to_numeric(maf.get('t_depth', 0), errors='coerce').fillna(0)
            maf['t_alt_count'] = pd.to_numeric(maf.get('t_alt_count', 0), errors='coerce').fillna(0)
            maf['VAF'] = np.where(maf['t_depth'] > 0, maf['t_alt_count'] / maf['t_depth'], 0)
            maf['SIFT'] = maf.get('SIFT', '').astype(str).str.lower()
            maf['PolyPhen'] = maf.get('PolyPhen', '').astype(str).str.lower()
            maf['DOMAINS'] = maf.get('DOMAINS', '').astype(str)
            maf['CONTEXT'] = maf.get('CONTEXT', '').astype(str).str.upper()

            def summarize(x):
                n = len(x)
                vc = x.get('Variant_Classification', pd.Series(index=x.index, dtype=object)).astype(str)
                vt = x.get('Variant_Type', pd.Series(index=x.index, dtype=object)).astype(str)
                pos_counts = x['Protein_Pos_Num'].dropna().astype(int).value_counts()
                hot_max = int(pos_counts.iloc[0]) if not pos_counts.empty else 0
                recurrent_positions = int((pos_counts >= 2).sum()) if not pos_counts.empty else 0
                protein_span = max(float(x['Protein_Pos_Num'].max() - x['Protein_Pos_Num'].min()), 1.0) if x['Protein_Pos_Num'].notna().any() else 1.0
                domain_mask = x['DOMAINS'].notna() & ~x['DOMAINS'].isin(['', '.', 'nan', 'NAN'])
                context = x['CONTEXT'].replace(['.', 'NAN', 'NONE'], np.nan).dropna()
                tri = context.str.slice(context.str.len() // 2 - 1, context.str.len() // 2 + 2)
                cpg = context.str.contains('CG', na=False).sum()
                ct_ctx = context.str.contains('C.T|G.A', regex=True, na=False).sum()
                deleterious = (x['SIFT'].str.contains('deleterious', na=False) | x['PolyPhen'].str.contains('probably|possibly|damaging', regex=True, na=False)).sum()
                high = vc.isin(['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']).sum()
                return pd.Series({'Hotspot_Max_Same_Position': hot_max, 'Hotspot_Recurrent_Position_Count': recurrent_positions, 'Hotspot_Fraction': hot_max / max(n, 1), 'Protein_Position_Cluster_Score': recurrent_positions / np.sqrt(protein_span + 1), 'Domain_Mutation_Count': int(domain_mask.sum()), 'Domain_Mutation_Ratio': float(domain_mask.mean()) if n else 0.0, 'Domain_Unique_Count': int(x.loc[domain_mask, 'DOMAINS'].nunique()), 'SIFT_Deleterious_Count': int(x['SIFT'].str.contains('deleterious', na=False).sum()), 'PolyPhen_Damaging_Count': int(x['PolyPhen'].str.contains('probably|possibly|damaging', regex=True, na=False).sum()), 'Deleterious_Impact_Ratio': deleterious / max(n, 1), 'Mean_Tumor_VAF': float(x['VAF'].mean()), 'Max_Tumor_VAF': float(x['VAF'].max()), 'Mutation_Context_Entropy': entropy(tri), 'CpG_Context_Ratio': cpg / max(len(context), 1), 'C_to_T_Context_Ratio': ct_ctx / max(len(context), 1), 'SNP_Fraction': (vt == 'SNP').sum() / max(n, 1), 'Indel_Fraction': vt.str.contains('INS|DEL', regex=True, na=False).sum() / max(n, 1), 'HighImpact_to_Context_Ratio': high / max(len(context), 1)})
            df = maf.groupby('Hugo_Symbol').apply(summarize).reset_index()
            df.rename(columns={'Hugo_Symbol': 'Gene'}, inplace=True)
            print(f'    Advanced mutation: {len(df)} genes')
            return df

        def get_oncoplex_mutation_recurrence_features(gene_list):
            path = resolve_data_file(ONCOPLEX_MUTATION_GENE_CANCER_FILE)
            if not path:
                return pd.DataFrame()
            print('  Loading ONCOPLEX cancer-wise mutation recurrence...')
            try:
                df = pd.read_csv(path, sep='\t', low_memory=False)
            except Exception as e:
                print(f'    ONCOPLEX recurrence skipped: {e}')
                return pd.DataFrame()
            if 'Hugo_Symbol' not in df.columns:
                return pd.DataFrame()
            genes = {str(g).upper() for g in gene_list}
            df['Gene'] = df['Hugo_Symbol'].astype(str).str.upper()
            df = df[df['Gene'].isin(genes)].copy()
            cancer_cols = [c for c in df.columns if c not in ['Hugo_Symbol', 'Gene']]
            if not cancer_cols or df.empty:
                print('    ONCOPLEX recurrence: 0 genes')
                return pd.DataFrame()
            mat = df[cancer_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            out = pd.DataFrame({'Gene': df['Gene'].values, 'OncoPlex_Mutation_MaxCancerRate': mat.max(axis=1).values, 'OncoPlex_Mutation_MeanCancerRate': mat.mean(axis=1).values, 'OncoPlex_Mutation_StdCancerRate': mat.std(axis=1).values, 'OncoPlex_Mutation_CancerTypes': (mat > 0).sum(axis=1).values, 'OncoPlex_PanCancer_Mutation_Burden': mat.sum(axis=1).values, 'OncoPlex_Mutation_TissueSpecificity': (mat.max(axis=1) / (mat.sum(axis=1) + 1e-06)).values})
            print(f'    ONCOPLEX recurrence: {len(out)} genes')
            return out

        def get_intogen_driver_features(gene_list):
            comp_path = resolve_data_file(INTOGEN_COMPENDIUM_FILE)
            unf_path = resolve_data_file(INTOGEN_UNFILTERED_FILE)
            if not comp_path and (not unf_path):
                return pd.DataFrame()
            print('  Loading IntOGen driver catalog features...')
            genes = {str(g).upper() for g in gene_list}
            method_names = ['dndscv', 'cbase', 'mutpanning', 'oncodrivefml', 'oncodriveclustl', 'hotmaps', 'smregions', 'combination']

            def method_flags(series, prefix):
                rows = []
                for text in series.fillna('').astype(str).str.lower():
                    parts = {p.strip() for p in text.replace(';', ',').split(',') if p.strip()}
                    rows.append({f'{prefix}_{m}': int(m in parts) for m in method_names})
                return pd.DataFrame(rows, index=series.index)
            frames = []
            if comp_path:
                comp = pd.read_csv(comp_path, sep='\t', low_memory=False)
                if 'SYMBOL' in comp.columns:
                    comp['Gene'] = comp['SYMBOL'].astype(str).str.upper()
                    comp = comp[comp['Gene'].isin(genes)].copy()
                if not comp.empty:
                    for col in ['MUTATIONS', 'SAMPLES', '%_SAMPLES_COHORT', 'QVALUE_COMBINATION', 'EXCESS_MIS', 'EXCESS_NON', 'EXCESS_SPL', 'TOTAL_SAMPLES']:
                        if col in comp.columns:
                            comp[col] = pd.to_numeric(comp[col], errors='coerce')
                    comp['Domain_Hit'] = comp.get('DOMAINS', '').notna() & ~comp.get('DOMAINS', '').astype(str).isin(['', 'nan', 'NaN'])
                    comp['Cluster2D_Hit'] = comp.get('2D_CLUSTERS', '').notna() & ~comp.get('2D_CLUSTERS', '').astype(str).isin(['', 'nan', 'NaN'])
                    comp['Cluster3D_Hit'] = comp.get('3D_CLUSTERS', '').notna() & ~comp.get('3D_CLUSTERS', '').astype(str).isin(['', 'nan', 'NaN'])
                    mf = method_flags(comp.get('METHODS', pd.Series('', index=comp.index)), 'IntOGen_Method')
                    comp = pd.concat([comp, mf], axis=1)
                    agg_map = {'COHORT': 'nunique', 'CANCER_TYPE': 'nunique', 'MUTATIONS': ['sum', 'max'], 'SAMPLES': ['sum', 'max'], '%_SAMPLES_COHORT': 'max', 'QVALUE_COMBINATION': 'min', 'EXCESS_MIS': 'max', 'EXCESS_NON': 'max', 'EXCESS_SPL': 'max', 'Domain_Hit': 'sum', 'Cluster2D_Hit': 'sum', 'Cluster3D_Hit': 'sum'}
                    for c in mf.columns:
                        agg_map[c] = 'max'
                    ca = comp.groupby('Gene').agg(agg_map)
                    ca.columns = ['_'.join([str(x) for x in col if str(x)]) for col in ca.columns]
                    ca = ca.reset_index().rename(columns={'COHORT_nunique': 'IntOGen_Driver_Cohort_Count', 'CANCER_TYPE_nunique': 'IntOGen_Driver_CancerType_Count', 'MUTATIONS_sum': 'IntOGen_Total_Mutations', 'MUTATIONS_max': 'IntOGen_Max_Cohort_Mutations', 'SAMPLES_sum': 'IntOGen_Total_Mutated_Samples', 'SAMPLES_max': 'IntOGen_Max_Cohort_Samples', '%_SAMPLES_COHORT_max': 'IntOGen_Max_Percent_Samples', 'QVALUE_COMBINATION_min': 'IntOGen_Min_QValue', 'EXCESS_MIS_max': 'IntOGen_Max_Excess_Mis', 'EXCESS_NON_max': 'IntOGen_Max_Excess_Non', 'EXCESS_SPL_max': 'IntOGen_Max_Excess_Spl', 'Domain_Hit_sum': 'IntOGen_Domain_Hit_Count', 'Cluster2D_Hit_sum': 'IntOGen_2D_Cluster_Count', 'Cluster3D_Hit_sum': 'IntOGen_3D_Cluster_Count'})
                    method_cols = [c for c in ca.columns if c.startswith('IntOGen_Method_')]
                    ca['IntOGen_Method_Count'] = ca[method_cols].sum(axis=1) if method_cols else 0
                    ca['IntOGen_Is_Compendium_Driver'] = 1
                    frames.append(ca)
            if unf_path:
                unf = pd.read_csv(unf_path, sep='\t', low_memory=False)
                if 'SYMBOL' in unf.columns:
                    unf['Gene'] = unf['SYMBOL'].astype(str).str.upper()
                    unf = unf[unf['Gene'].isin(genes)].copy()
                if not unf.empty:
                    for col in ['MUTATIONS', 'SAMPLES_COHORT', 'QVALUE_COMBINATION', 'RANKING', 'TIER', 'SAMPLES_3MUTS', 'NUM_PAPERS']:
                        if col in unf.columns:
                            unf[col] = pd.to_numeric(unf[col], errors='coerce')
                    sig = method_flags(unf.get('SIG_METHODS', pd.Series('', index=unf.index)), 'IntOGen_SigMethod')
                    unf = pd.concat([unf, sig], axis=1)
                    ua = unf.groupby('Gene').agg({'COHORT': 'nunique', 'CANCER_TYPE': 'nunique', 'QVALUE_COMBINATION': 'min', 'RANKING': 'min', 'TIER': 'min', 'SAMPLES_3MUTS': 'sum', 'NUM_PAPERS': 'max', 'DRIVER': lambda x: pd.Series(x).astype(str).str.lower().isin(['true', '1']).sum(), 'WARNING_ARTIFACT': lambda x: pd.Series(x).astype(str).str.lower().isin(['true', '1']).sum(), 'KNOWN_ARTIFACT': lambda x: pd.Series(x).astype(str).str.lower().isin(['true', '1']).sum()}).reset_index().rename(columns={'COHORT': 'IntOGen_Unfiltered_Cohort_Count', 'CANCER_TYPE': 'IntOGen_Unfiltered_CancerType_Count', 'QVALUE_COMBINATION': 'IntOGen_Unfiltered_Min_QValue', 'RANKING': 'IntOGen_Best_Ranking', 'TIER': 'IntOGen_Best_Tier', 'SAMPLES_3MUTS': 'IntOGen_Samples_3Muts', 'NUM_PAPERS': 'IntOGen_CancerMine_NumPapers', 'DRIVER': 'IntOGen_Unfiltered_Driver_Count', 'WARNING_ARTIFACT': 'IntOGen_Warning_Artifact_Count', 'KNOWN_ARTIFACT': 'IntOGen_Known_Artifact_Count'})
                    sig_cols = sig.columns.tolist()
                    sig_agg = pd.concat([unf[['Gene']], sig], axis=1).groupby('Gene').max().reset_index()
                    ua = ua.merge(sig_agg, on='Gene', how='left')
                    ua['IntOGen_SigMethod_Count'] = ua[sig_cols].sum(axis=1) if sig_cols else 0
                    frames.append(ua)
            if not frames:
                print('    IntOGen: 0 genes')
                return pd.DataFrame()
            out = frames[0]
            for fr in frames[1:]:
                out = out.merge(fr, on='Gene', how='outer')
            out = out.fillna(0)
            print(f'    IntOGen: {len(out)} genes')
            return out

        def get_cnv_features(gene_list):
            if not E(CNV_FILE):
                return pd.DataFrame()
            print('  Loading CNV...')
            cnv = pd.read_csv(CNV_FILE, sep='\t', compression='gzip', index_col=0)
            cnv = cnv.apply(pd.to_numeric, errors='coerce')
            cnv = cnv[cnv.index.isin(set(gene_list))]
            df = pd.DataFrame({'Gene': cnv.index, 'CNV_Mean': cnv.mean(axis=1).values, 'CNV_STD': cnv.std(axis=1).values, 'CNV_Gain_Ratio': (cnv > 0).mean(axis=1).values, 'CNV_Loss_Ratio': (cnv < 0).mean(axis=1).values, 'CNV_Amplification_Ratio': (cnv >= 1).mean(axis=1).values, 'CNV_Deletion_Ratio': (cnv <= -1).mean(axis=1).values})
            print(f'    CNV: {len(df)} genes')
            return df

        def get_expression_features(gene_list):
            if not E(EXPR_FILE):
                return pd.DataFrame()
            print('  Loading expression...')
            expr = pd.read_csv(EXPR_FILE, sep='\t', compression='gzip', index_col=0)
            expr = expr.apply(pd.to_numeric, errors='coerce')
            expr = expr[expr.index.isin(set(gene_list))]
            df = pd.DataFrame({'Gene': expr.index, 'Expr_Mean': expr.mean(axis=1).values, 'Expr_STD': expr.std(axis=1).values, 'Expr_Median': expr.median(axis=1).values, 'Expr_CV': (expr.std(axis=1) / (expr.mean(axis=1).abs() + 1e-06)).values})
            print(f'    Expression: {len(df)} genes')
            return df

        def get_clinvar_features(gene_list):
            if not E(CLINVAR_FILE):
                return pd.DataFrame()
            print('  Loading ClinVar...')
            cv = pd.read_csv(CLINVAR_FILE, sep='\t', low_memory=False)
            cv = cv[cv['GeneSymbol'].isin(set(gene_list))]
            cv['sig'] = cv['ClinicalSignificance'].str.lower()
            cv['typ'] = cv['Type'].str.lower()
            cv['rev'] = cv['ReviewStatus'].str.lower()
            df = cv.groupby('GeneSymbol').agg(ClinVar_Total_Variants=('VariationID', 'count'), ClinVar_Pathogenic_Count=('sig', lambda x: x.str.contains('pathogenic', na=False).sum()), ClinVar_Benign_Count=('sig', lambda x: x.str.contains('benign', na=False).sum()), ClinVar_Uncertain_Count=('sig', lambda x: x.str.contains('uncertain', na=False).sum()), ClinVar_Reviewed_Pathogenic=('rev', lambda x: x.str.contains('expert|guideline', na=False).sum()), ClinVar_SNV_Count=('typ', lambda x: x.str.contains('single nucleotide', na=False).sum()), ClinVar_Indel_Count=('typ', lambda x: x.str.contains('deletion|insertion', na=False).sum()), ClinVar_Structural_Count=('typ', lambda x: x.str.contains('duplication|copy|structural', na=False).sum())).reset_index()
            df.rename(columns={'GeneSymbol': 'Gene'}, inplace=True)
            df['ClinVar_Pathogenic_Ratio'] = df['ClinVar_Pathogenic_Count'] / df['ClinVar_Total_Variants'].clip(1)
            print(f'    ClinVar: {len(df)} genes')
            return df

        def get_pathway_features(gene_list):
            if not E(PATHWAY_FILE):
                return pd.DataFrame()
            print('  Loading pathways...')
            gset = set((g.upper() for g in gene_list))
            g2p = defaultdict(set)
            with open(PATHWAY_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    for g in parts[2:]:
                        g = g.strip().upper()
                        if g in gset:
                            g2p[g].add(parts[0])
            df = pd.DataFrame({'Gene': list(g2p.keys()), 'pathway_count_all': [len(v) for v in g2p.values()]})
            print(f'    Pathway: {len(df)} genes')
            return df

        def get_ppi_features(gene_list, cgc_genes=None):
            links_path = resolve_data_file(PPI_LINKS)
            aliases_path = resolve_data_file(PPI_ALIASES)
            if not links_path or not aliases_path:
                print('    PPI: files not found — skip')
                return pd.DataFrame()
            gset = set((g.upper() for g in gene_list))
            cgc_set = set((g.upper() for g in cgc_genes or []))
            tmpdir = tempfile.gettempdir()

            def safe_read(path, **kw):
                tmp = os.path.join(tmpdir, '_ppi_' + os.path.basename(path))
                try:
                    shutil.copy2(path, tmp)
                    df = pd.read_csv(tmp, **kw)
                    return df
                except PermissionError:
                    return pd.read_csv(path, **kw)
                finally:
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except:
                        pass
            try:
                print('  Loading PPI (STRING)...')
                print(f'    aliases: {aliases_path}')
                print(f'    links  : {links_path}')
                alias = safe_read(aliases_path, sep='\t', low_memory=False)
                alias.columns = alias.columns.str.strip()
                prot_col = [c for c in alias.columns if 'protein' in c.lower()][0]
                alias_col = [c for c in alias.columns if 'alias' in c.lower()][0]
                src_col = [c for c in alias.columns if 'source' in c.lower()][0]
                alias[prot_col] = alias[prot_col].astype(str).str.strip()
                alias[alias_col] = alias[alias_col].astype(str).str.strip()
                alias[src_col] = alias[src_col].astype(str)
                pref = alias[alias[src_col].str.contains('HGNC|Gene_Name|BioMart_HUGO|Ensembl_HGNC', na=False, case=False)]
                found = set(pref[alias_col].str.upper())
                miss = gset - found
                if len(miss) > len(gset) * 0.3:
                    fb = alias[alias[alias_col].str.upper().isin(miss)]
                    combined = pd.concat([pref, fb]).drop_duplicates(prot_col)
                else:
                    combined = pref
                prot2gene = {}
                for _, row in combined.sort_values(src_col).iterrows():
                    pid = str(row[prot_col])
                    gene = str(row[alias_col]).upper()
                    if pid not in prot2gene:
                        prot2gene[pid] = gene
                links = safe_read(links_path, sep=' ', low_memory=False)
                links.columns = links.columns.str.strip()
                links = links[links['combined_score'] >= 700]
                links['protein1'] = links['protein1'].astype(str).str.strip()
                links['protein2'] = links['protein2'].astype(str).str.strip()
                links['g1'] = links['protein1'].map(prot2gene)
                links['g2'] = links['protein2'].map(prot2gene)
                links = links.dropna(subset=['g1', 'g2'])
                sub = links[links['g1'].isin(gset) | links['g2'].isin(gset)]
                d1 = sub.groupby('g1')['g2'].count().rename('PPI_degree')
                d2 = sub.groupby('g2')['g1'].count().rename('PPI_degree')
                ppi = pd.concat([d1, d2]).groupby(level=0).sum()
                if cgc_set:
                    cn1 = sub[sub['g2'].isin(cgc_set)].groupby('g1').size().rename('CGC_neighbor_count')
                    cn2 = sub[sub['g1'].isin(cgc_set)].groupby('g2').size().rename('CGC_neighbor_count')
                    cgc_nb = pd.concat([cn1, cn2]).groupby(level=0).sum()
                else:
                    cgc_nb = pd.Series(dtype=float, name='CGC_neighbor_count')
                result = pd.concat([ppi, cgc_nb], axis=1).fillna(0).reset_index()
                result.columns = ['Gene', 'PPI_degree', 'CGC_neighbor_count']
                result = result[result['Gene'].isin(gset)].copy()
                result['PPI_degree'] = result['PPI_degree'].astype(int)
                result['CGC_neighbor_count'] = result['CGC_neighbor_count'].astype(int)
                all_genes = pd.DataFrame({'Gene': sorted(gset)})
                result = all_genes.merge(result, on='Gene', how='left').fillna(0)
                result['PPI_degree'] = result['PPI_degree'].astype(int)
                result['CGC_neighbor_count'] = result['CGC_neighbor_count'].astype(int)
                nonzero = int((result['PPI_degree'] > 0).sum())
                cgc_nonzero = int((result['CGC_neighbor_count'] > 0).sum())
                print(f'    PPI: {len(result)} genes, {nonzero} with degree, {cgc_nonzero} with CGC neighbours')
                return result
            except Exception as e:
                print(f'    PPI ERROR: {e} — skip')
                return pd.DataFrame()

        def fill_existing_network_zeros(base, cgc_genes):
            for col in ['PPI_degree', 'CGC_neighbor_count']:
                base[col] = pd.to_numeric(base[col], errors='coerce').fillna(0)
            zero_mask = (base['PPI_degree'] <= 0) | (base['CGC_neighbor_count'] <= 0)
            zero_genes = sorted(base.loc[zero_mask, 'Gene'].astype(str).str.upper().unique())
            print(f'\n  Existing network zeros to check: {len(zero_genes)} genes')
            print(f"    PPI_degree zero/missing        : {int((base['PPI_degree'] <= 0).sum())}")
            print(f"    CGC_neighbor_count zero/missing: {int((base['CGC_neighbor_count'] <= 0).sum())}")
            if not zero_genes:
                return base
            ppi_fix = get_ppi_features(zero_genes, cgc_genes)
            if ppi_fix is None or ppi_fix.empty:
                print('  Existing network zeros not updated because PPI extraction returned empty.')
                return base
            ppi_fix = ppi_fix.set_index('Gene')
            before_ppi = int((base['PPI_degree'] <= 0).sum())
            before_cgc = int((base['CGC_neighbor_count'] <= 0).sum())
            for idx, row in base.iterrows():
                gene = str(row['Gene']).upper()
                if gene not in ppi_fix.index:
                    continue
                new_ppi = int(ppi_fix.at[gene, 'PPI_degree'])
                new_cgc = int(ppi_fix.at[gene, 'CGC_neighbor_count'])
                if row['PPI_degree'] <= 0 and new_ppi > 0:
                    base.at[idx, 'PPI_degree'] = new_ppi
                if row['CGC_neighbor_count'] <= 0 and new_cgc > 0:
                    base.at[idx, 'CGC_neighbor_count'] = new_cgc
            after_ppi = int((base['PPI_degree'] <= 0).sum())
            after_cgc = int((base['CGC_neighbor_count'] <= 0).sum())
            print(f'  Existing PPI_degree fixed        : {before_ppi - after_ppi}')
            print(f'  Existing CGC_neighbor_count fixed: {before_cgc - after_cgc}')
            return base

        def oversample_positive_rows(df, target_ratio=1.0, seed=42):
            df = df.copy()
            df['Label'] = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
            pos_count = int((df['Label'] == 1).sum())
            neg_count = int((df['Label'] == 0).sum())
            target_pos = int(np.ceil(neg_count / target_ratio))
            need = max(0, target_pos - pos_count)
            if need == 0:
                print('  Oversampling not needed; dataset already meets target ratio.')
                return df
            pos_rows = df[df['Label'] == 1].copy()
            if pos_rows.empty:
                print('  Oversampling skipped; no Label=1 rows found.')
                return df
            add = pos_rows.sample(n=need, replace=True, random_state=seed).copy().reset_index(drop=True)
            if 'Gene' in add.columns:
                add['Gene'] = add['Gene'].astype(str) + [f'__AUG{i + 1:04d}' for i in range(len(add))]
            balanced = pd.concat([df, add], ignore_index=True)
            print(f'  Oversampled Label=1 rows: {need}')
            print(f"  Balanced counts after oversampling: {int((balanced['Label'] == 1).sum())} pos / {int((balanced['Label'] == 0).sum())} neg")
            return balanced

        def get_methylation_features(gene_list):
            if not E(METH_FILE) or not E(MANIFEST):
                print('    Methylation: files not found — skip')
                return pd.DataFrame()
            gset = set((g.upper() for g in gene_list))
            print('  Loading methylation...')
            try:
                hrow = 0
                with open(MANIFEST, 'r', encoding='latin1', errors='ignore') as f:
                    for i, line in enumerate(f):
                        if 'IlmnID' in line or 'ilmnid' in line.lower():
                            hrow = i
                            break
                mf = pd.read_csv(MANIFEST, skiprows=hrow, low_memory=False, encoding='latin1')
                mf.columns = mf.columns.str.strip().str.lower()
                pc = next((c for c in mf.columns if 'ilmnid' in c.lower() or c == 'name'), None)
                gc = next((c for c in mf.columns if 'refgene_name' in c.lower()), None)
                grpc = next((c for c in mf.columns if 'refgene_group' in c.lower()), None)
                if not pc or not gc:
                    print('    Manifest columns missing')
                    return pd.DataFrame()
                mf[gc] = mf[gc].fillna('').astype(str)
                if grpc:
                    mf[grpc] = mf[grpc].fillna('').astype(str)
                    pm = mf[mf[grpc].str.contains('TSS200|TSS1500', case=False, na=False)][[pc, gc]].copy()
                else:
                    pm = mf[[pc, gc]].copy()
                pm = pm[pm[gc] != ''].copy()
                pm[gc] = pm[gc].str.split(';')
                pm = pm.explode(gc)
                pm[gc] = pm[gc].str.strip().str.upper()
                pm = pm[pm[gc].isin(gset)]
                pm.columns = ['probe', 'gene']
                pm = pm.dropna().reset_index(drop=True)
                if len(pm) == 0:
                    print('    No probes matched')
                    return pd.DataFrame()
                print(f"    Probe map: {len(pm):,} pairs, {pm['gene'].nunique()} genes")
            except Exception as e:
                print(f'    Manifest error: {e}')
                return pd.DataFrame()
            gs, gq, gc_d, ghy, ghypo = ({}, {}, {}, {}, {})
            nc = 0
            for chunk in pd.read_csv(METH_FILE, sep='\t', compression='gzip', chunksize=1000, index_col=0):
                merged = chunk.merge(pm, left_index=True, right_on='probe', how='inner')
                if merged.empty:
                    nc += 1
                    continue
                bc = [c for c in merged.columns if c not in ('probe', 'gene')]
                pv = merged[bc].apply(pd.to_numeric, errors='coerce').mean(axis=1)
                for g, v in zip(merged['gene'], pv):
                    if pd.isna(v):
                        continue
                    v = float(v)
                    gs[g] = gs.get(g, 0.0) + v
                    gq[g] = gq.get(g, 0.0) + v * v
                    gc_d[g] = gc_d.get(g, 0) + 1
                    ghy[g] = ghy.get(g, 0) + (1 if v > 0.6 else 0)
                    ghypo[g] = ghypo.get(g, 0) + (1 if v < 0.3 else 0)
                nc += 1
                if nc % 100 == 0:
                    print(f'      meth chunk {nc}, genes: {len(gs)}')
            if not gs:
                print('    No methylation data found')
                return pd.DataFrame()
            genes = list(gs.keys())
            df = pd.DataFrame({'Gene': genes, 'Meth_Mean': [gs[g] / gc_d[g] for g in genes], 'Meth_STD': [np.sqrt(max(gq[g] / gc_d[g] - (gs[g] / gc_d[g]) ** 2, 0)) for g in genes], 'Meth_Hyper_Ratio': [ghy[g] / gc_d[g] for g in genes], 'Meth_Hypo_Ratio': [ghypo[g] / gc_d[g] for g in genes], 'Prom_Meth_Mean': [gs[g] / gc_d[g] for g in genes], 'Prom_Meth_STD': [np.sqrt(max(gq[g] / gc_d[g] - (gs[g] / gc_d[g]) ** 2, 0)) for g in genes], 'Prom_Meth_Probe_Count': [gc_d[g] for g in genes]})
            print(f'    Methylation: {len(df)} genes')
            return df

        def fill_existing_methylation_zeros(base):
            for col in METHYLATION_COLS:
                base[col] = pd.to_numeric(base[col], errors='coerce').fillna(0)
            zero_mask = (base['Meth_Mean'] <= 0) | (base['Prom_Meth_Mean'] <= 0) | (base['Prom_Meth_Probe_Count'] <= 0)
            zero_genes = sorted(base.loc[zero_mask, 'Gene'].astype(str).str.upper().unique())
            print(f'\n  Existing methylation zeros to check: {len(zero_genes)} genes')
            for col in METHYLATION_COLS:
                print(f'    {col:<22s}: {int((base[col] <= 0).sum())} zero/missing')
            if not zero_genes:
                return base
            meth_fix = get_methylation_features(zero_genes)
            if meth_fix is None or meth_fix.empty:
                print('  Existing methylation zeros not updated because extraction returned empty.')
                return base
            meth_fix['Gene'] = meth_fix['Gene'].astype(str).str.upper()
            meth_fix = meth_fix.set_index('Gene')
            before = {col: int((base[col] <= 0).sum()) for col in METHYLATION_COLS}
            for idx, row in base.iterrows():
                gene = str(row['Gene']).upper()
                if gene not in meth_fix.index:
                    continue
                for col in METHYLATION_COLS:
                    new_val = pd.to_numeric(meth_fix.at[gene, col], errors='coerce')
                    if pd.notna(new_val) and row[col] <= 0 and (float(new_val) > 0):
                        base.at[idx, col] = float(new_val)
            after = {col: int((base[col] <= 0).sum()) for col in METHYLATION_COLS}
            for col in METHYLATION_COLS:
                print(f'  Existing {col} fixed: {before[col] - after[col]}')
            if 'Prom_Meth_Probe_Count' in base.columns:
                base['Prom_Meth_Probe_Count'] = base['Prom_Meth_Probe_Count'].round().astype(int)
            return base

        def fetch_with_checkpoint(gene_label_list, checkpoint_file):
            requested = {str(g).upper(): label for g, label in gene_label_list}
            done = set()
            cp_rows = []
            seq_rows = []
            if os.path.exists(checkpoint_file):
                cp = pd.read_csv(checkpoint_file)
                if 'Gene' in cp.columns:
                    cp['Gene'] = cp['Gene'].astype(str).str.upper()
                    cp = cp[cp['Gene'].isin(requested)].copy()
                done = {str(r['Gene']).upper() for _, r in cp.iterrows()}
                cp_rows = cp.to_dict('records')
                seq_rows = [r for r in cp_rows if r.get('Label', -999) != -999 and requested.get(str(r.get('Gene', '')).upper()) == r.get('Label')]
                print(f'  Checkpoint loaded: {len(done)} genes done')
            failed = []
            total = len(gene_label_list)
            for i, (gene, label) in enumerate(gene_label_list, 1):
                gup = gene.upper()
                if gup in done:
                    print(f'  [{i:3d}/{total}] {gup:<12s}  [checkpoint]')
                    continue
                tag = 'POSITIVE' if label == 1 else 'negative'
                print(f'  [{i:3d}/{total}] {gup:<12s} [{tag}]', end='  ')
                hdr, seq = fetch_ncbi(gup)
                if seq:
                    feat = seq_features(gup, seq)
                    feat['Label'] = label
                    seq_rows.append(feat)
                    cp_rows.append({**feat, 'Label': label})
                    done.add(gup)
                    print(f'✓ ({len(seq):,} nt)')
                else:
                    failed.append(gup)
                    cp_rows.append({'Gene': gup, 'Label': -999})
                    print('✗')
                if i % 50 == 0:
                    pd.DataFrame(cp_rows).to_csv(checkpoint_file, index=False)
                    print(f'  → Checkpoint saved ({i}/{total})')
            pd.DataFrame(cp_rows).to_csv(checkpoint_file, index=False)
            print(f'  Checkpoint final save → {checkpoint_file}')
            return (seq_rows, failed)

        def main():
            if not EMAIL.strip():
                raise ValueError('Set NCBI_EMAIL to a valid contact email before sequence extraction.')
            print('=' * 64)
            print('build_balanced_dataset.py — Script 1')
            print('=' * 64)
            base = pd.read_csv(INPUT_CSV)
            base.columns = base.columns.str.strip()
            base['Gene'] = base['Gene'].str.upper()
            base['Label'] = pd.to_numeric(base['Label'], errors='coerce').fillna(0).astype(int)
            base = drop_derived_proxy_features(base)
            existing_all = set(base['Gene'])
            existing_pos = set(base[base['Label'] == 1]['Gene'])
            for net_col in ['PPI_degree', 'CGC_neighbor_count']:
                if net_col not in base.columns:
                    insert_at = base.columns.get_loc('Label') if 'Label' in base.columns else len(base.columns)
                    base.insert(insert_at, net_col, 0)
            for meth_col in METHYLATION_COLS:
                if meth_col not in base.columns:
                    insert_at = base.columns.get_loc('Label') if 'Label' in base.columns else len(base.columns)
                    base.insert(insert_at, meth_col, 0)
            existing_cols = list(base.columns)
            print(f'\n  Input     : {INPUT_CSV}')
            print(f'  Genes     : {len(base)}')
            print(f'  Label=1   : {len(existing_pos)}')
            print(f"  Label=0   : {(base['Label'] == 0).sum()}")
            print(f"  Imbalance : {(base['Label'] == 0).sum() / len(existing_pos):.2f}:1")
            all_pos_unique = list(dict.fromkeys((g.upper() for g in ALL_NEW_POSITIVE)))
            all_neg_unique = list(dict.fromkeys((g.upper() for g in NCG_CANDIDATES_NEG + TUMOR_SUPPRESSORS_NEG + HOUSEKEEPING_NEG)))
            user_labeled_fetch = load_user_labeled_genes(USER_GENE_LABELS_FILE, existing_all)
            cgc_fetch = load_cgc_positive_genes(CGC_FILE, existing_all)
            extra_pos_fetch = load_extra_positive_genes(EXTRA_POSITIVE_GENES_FILE, existing_all)
            oncoplex_pos_fetch = load_oncoplex_gene_list(ONCOPLEX_POSITIVE_FILE, 1, existing_all)
            oncoplex_neg_fetch = load_oncoplex_gene_list(ONCOPLEX_NEGATIVE_FILE, 0, existing_all)
            oncoplex_npg_fetch = load_oncoplex_npg_genes(ONCOPLEX_NPG_DIR, existing_all)
            additional_driver_fetch = load_additional_driver_gene_sets(ADDITIONAL_GENE_SETS_DIR, existing_all)
            relabel = set(all_pos_unique) & existing_all - existing_pos
            if relabel:
                print(f'\n  Relabelling {len(relabel)} genes: 0→1')
                print(f'  {sorted(relabel)}')
                base.loc[base['Gene'].isin(relabel), 'Label'] = 1
                existing_pos.update(relabel)
            if E(USER_GENE_LABELS_FILE):
                try:
                    combined_labels = pd.read_csv(USER_GENE_LABELS_FILE)
                    if {'Gene', 'Label', 'Conflict_NegativeWins'}.issubset(combined_labels.columns):
                        combined_labels['Gene'] = combined_labels['Gene'].astype(str).str.upper().str.strip()
                        combined_labels['Label'] = pd.to_numeric(combined_labels['Label'], errors='coerce')
                        combined_labels['Conflict_NegativeWins'] = pd.to_numeric(combined_labels['Conflict_NegativeWins'], errors='coerce').fillna(0).astype(int)
                        conflict_neg = set(combined_labels.loc[(combined_labels['Label'] == 0) & (combined_labels['Conflict_NegativeWins'] == 1), 'Gene'])
                        relabel_neg = conflict_neg & existing_pos
                        if relabel_neg:
                            print(f'\n  Relabelling {len(relabel_neg)} conflict genes: 1→0')
                            print(f'  {sorted(relabel_neg)}')
                            base.loc[base['Gene'].isin(relabel_neg), 'Label'] = 0
                            existing_pos.difference_update(relabel_neg)
                except Exception as e:
                    print(f'  Conflict-negative relabel skipped: {e}')
            if os.path.exists(BASE_REPAIR_CACHE):
                cached_base = pd.read_csv(BASE_REPAIR_CACHE)
                cached_base.columns = cached_base.columns.str.strip()
                cached_base = drop_derived_proxy_features(cached_base)
                if 'Gene' in cached_base.columns and len(cached_base) == len(base):
                    cached_base['Gene'] = cached_base['Gene'].astype(str).str.upper()
                    if set(cached_base['Gene']) == set(base['Gene']):
                        print(f'\n  Base repair cache loaded: {BASE_REPAIR_CACHE}')
                        base = cached_base
                    else:
                        cached_base = None
                else:
                    cached_base = None
            else:
                cached_base = None
            if cached_base is None:
                if RECOMPUTE_EXISTING_NETWORK_ZEROS:
                    base = fill_existing_network_zeros(base, existing_pos)
                    safe_to_csv(base, BASE_REPAIR_CACHE, index=False)
                if RECOMPUTE_EXISTING_METHYLATION_ZEROS:
                    base = fill_existing_methylation_zeros(base)
                    safe_to_csv(base, BASE_REPAIR_CACHE, index=False)
            before_source_cols = set(base.columns)
            for feat_fn, name in [(get_oncoplex_mutation_recurrence_features, 'ONCOPLEX recurrence'), (get_intogen_driver_features, 'IntOGen driver catalog'), (get_mutation_context_domain_features, 'MAF context/domain')]:
                try:
                    fdf = feat_fn(base['Gene'].tolist())
                    base = merge_feature_frame(base, fdf)
                except Exception as e:
                    print(f'  Existing {name} features skipped: {e}')
            source_added = [c for c in base.columns if c not in before_source_cols]
            if source_added:
                print(f'\n  Source-derived driver features added: {len(source_added)}')
                print(f'  Source feature sample: {source_added[:8]}')
            if ADD_ADVANCED_DRIVER_FEATURES:
                before_cols = set(base.columns)
                base = add_advanced_driver_features(base)
                base = move_label_to_end(base)
                added_cols = [c for c in base.columns if c not in before_cols]
                print(f'\n  Advanced driver features added: {len(added_cols)}')
                if added_cols:
                    print(f'  Advanced feature sample: {added_cols[:8]}')
                existing_cols = list(base.columns)
            else:
                base = drop_derived_proxy_features(base)
                base = move_label_to_end(base)
                existing_cols = list(base.columns)
            all_fetch_pos = [(g, 1) for g in all_pos_unique if g not in existing_all]
            all_fetch_pos += [(g, 1) for g, lbl in user_labeled_fetch if lbl == 1 and g not in existing_all]
            all_fetch_pos += cgc_fetch
            all_fetch_pos += extra_pos_fetch
            all_fetch_pos += oncoplex_pos_fetch
            all_fetch_pos += oncoplex_npg_fetch
            all_fetch_pos += additional_driver_fetch
            all_fetch_pos = list(dict.fromkeys(all_fetch_pos))
            current_pos = int((base['Label'] == 1).sum())
            current_neg = int((base['Label'] == 0).sum())
            target_pos = int(np.ceil(current_neg / TARGET_NEG_POS_RATIO))
            pos_needed = max(0, target_pos - current_pos)
            fetch_pos = all_fetch_pos if FETCH_ALL_AVAILABLE_POSITIVES else all_fetch_pos[:pos_needed]
            fetch_neg = [(g, 0) for g, lbl in user_labeled_fetch if lbl == 0 and g not in existing_all]
            fetch_neg += oncoplex_neg_fetch
            if ADD_NEGATIVE_INSTANCES:
                fetch_neg += [(g, 0) for g in all_neg_unique if g not in existing_all]
            fetch_neg = list(dict.fromkeys(fetch_neg))
            if pos_needed > len(all_fetch_pos):
                print(f'\n  Warning: need {pos_needed} Label=1 genes for target balance, but only {len(all_fetch_pos)} new positive candidates are available.')
            if ADD_NEGATIVE_INSTANCES and E(MAF_FILE):
                print('\n  Getting TCGA background genes...')
                m = pd.read_csv(MAF_FILE, sep='\t', compression='gzip', low_memory=False, usecols=['Hugo_Symbol', 'Tumor_Sample_Barcode', 'FILTER'])
                m = m[m['FILTER'] == 'PASS']
                gc_m = m.groupby('Hugo_Symbol')['Tumor_Sample_Barcode'].nunique()
                bg = gc_m[(gc_m >= 5) & (gc_m <= 50) & ~gc_m.index.isin(existing_pos) & ~gc_m.index.isin(existing_all)]
                bg = bg.sort_values().iloc[len(bg) // 4:3 * len(bg) // 4]
                bg_sel = bg.sample(min(300, max(len(bg) // 2, 1)), random_state=42)
                fetch_neg += [(g.upper(), 0) for g in bg_sel.index]
                print(f'  TCGA background: {len(bg_sel)} genes')
            planned_genes = {g for g, _ in fetch_pos + fetch_neg}
            fetch_neg += get_tcga_background_gene_labels(existing_all, set(all_pos_unique) | {g for g, lbl in user_labeled_fetch if lbl == 1} | {g for g, _ in cgc_fetch} | {g for g, _ in extra_pos_fetch} | {g for g, _ in oncoplex_pos_fetch} | {g for g, _ in oncoplex_npg_fetch} | {g for g, _ in additional_driver_fetch}, planned_genes)
            fetch_neg = list(dict.fromkeys(fetch_neg))
            max_new_neg = None
            if not FETCH_ALL_AVAILABLE_NEGATIVES:
                max_new_neg = max(0, int(np.ceil((current_pos + len(fetch_pos)) * TARGET_NEG_POS_RATIO)) - current_neg)
                if len(fetch_neg) > max_new_neg:
                    print(f'\n  Limiting Label=0 fetch for balance: {len(fetch_neg)} -> {max_new_neg}')
                fetch_neg = fetch_neg[:max_new_neg]
            all_to_fetch = list(dict.fromkeys(fetch_pos + fetch_neg))
            n_pos_fetch = len([x for x in all_to_fetch if x[1] == 1])
            n_neg_fetch = len([x for x in all_to_fetch if x[1] == 0])
            print(f"\n{'─' * 64}")
            print('  FETCH PLAN')
            print(f"{'─' * 64}")
            print(f'  Relabelled (0→1)  : {len(relabel)}')
            print(f'  Current Label=1   : {current_pos}')
            print(f'  Current Label=0   : {current_neg}')
            print(f'  Target ratio 0:1  : {TARGET_NEG_POS_RATIO:.2f}:1')
            print(f'  Label=1 needed    : {pos_needed}')
            print(f'  Fetch all pos     : {FETCH_ALL_AVAILABLE_POSITIVES}')
            print(f'  Fetch all neg     : {FETCH_ALL_AVAILABLE_NEGATIVES}')
            if max_new_neg is not None:
                print(f'  Label=0 cap       : {max_new_neg}')
            print(f'  New Label=1 fetch : {n_pos_fetch}')
            print(f'  New Label=0 fetch : {n_neg_fetch}')
            print(f"  CGC file          : {('enabled' if ADD_CGC_GENES else 'disabled')}")
            print(f"  Extra positives   : {(EXTRA_POSITIVE_GENES_FILE if ADD_EXTRA_POSITIVE_GENES else 'disabled')}")
            print(f"  ONCOPLEX lists    : {('enabled' if ADD_ONCOPLEX_GENE_LISTS else 'disabled')}")
            print(f"  ONCOPLEX NPGs     : {(ONCOPLEX_NPG_DIR if ADD_ONCOPLEX_NPG_GENES else 'disabled')}")
            print(f"  Additional drivers: {(ADDITIONAL_GENE_SETS_DIR if ADD_ADDITIONAL_DRIVER_GENE_SETS else 'disabled')}")
            print(f"  User labeled file : {(USER_GENE_LABELS_FILE if ADD_USER_LABELED_GENES else 'disabled')}")
            print(f"  TCGA background   : {('enabled' if ADD_TCGA_BACKGROUND_INSTANCES else 'disabled')}")
            print(f'  Oversample Label=1: {OVERSAMPLE_POSITIVES_TO_BALANCE}')
            print(f'  Total             : {len(all_to_fetch)}')
            print(f'  Checkpoint file   : {CHECKPOINT}')
            print(f'  Est. time         : ~{len(all_to_fetch) * 0.7 / 60:.0f} min')
            proj_pos = len(existing_pos) + n_pos_fetch
            proj_neg = (base['Label'] == 0).sum() + n_neg_fetch
            print(f'\n  Projected result  : ~{proj_pos} pos / ~{proj_neg} neg')
            print(f'  Projected ratio   : ~{proj_neg / proj_pos:.2f}:1  (was 2.88:1)')
            if all_to_fetch:
                print(f"\n{'─' * 64}")
                print('  FETCHING FROM NCBI (resume-safe)')
                print(f"{'─' * 64}")
                seq_rows, failed = fetch_with_checkpoint(all_to_fetch, CHECKPOINT)
            else:
                print('\n  No new genes needed for the requested balance target.')
                seq_rows, failed = ([], [])
            if failed:
                safe_to_csv(pd.DataFrame({'Gene': failed}), 'fetch_failed.csv', index=False)
            pos_rows = [r for r in seq_rows if r.get('Label') == 1]
            neg_rows = [r for r in seq_rows if r.get('Label') == 0]
            print(f'\n  Fetched Label=1 : {len(pos_rows)}')
            print(f'  Fetched Label=0 : {len(neg_rows)}')
            print(f'  Failed          : {len(failed)}')
            if not seq_rows:
                base['Label'] = pd.to_numeric(base['Label'], errors='coerce').fillna(0).astype(int)
                base = base.fillna(0)
                if OVERSAMPLE_POSITIVES_TO_BALANCE:
                    base = oversample_positive_rows(base, TARGET_NEG_POS_RATIO, RANDOM_STATE)
                base = move_label_to_end(base)
                saved_path = safe_to_csv(base, OUTPUT_CSV, index=False)
                print(f'  Nothing fetched. Saved relabel/network-fixed dataset → {saved_path}')
                return
            requested_genes = {str(r['Gene']).upper() for r in seq_rows if r.get('Label', -999) != -999}
            if os.path.exists(FEATURE_CACHE):
                cached = pd.read_csv(FEATURE_CACHE)
                cached['Gene'] = cached['Gene'].astype(str).str.upper()
                cached_genes = set(cached['Gene'])
                if requested_genes and requested_genes.issubset(cached_genes):
                    print(f'\n  Feature cache loaded: {FEATURE_CACHE}')
                    new_df = cached[cached['Gene'].isin(requested_genes)].copy()
                else:
                    print(f'\n  Feature cache found but does not match current fetch plan; rebuilding.')
                    new_df = None
            else:
                new_df = None
            if new_df is None:
                new_df = pd.DataFrame(seq_rows)
                new_df = new_df[new_df['Label'] != -999].copy()
                new_df['Gene'] = new_df['Gene'].astype(str).str.upper()
                fetched_genes = new_df['Gene'].tolist()
                all_cgc = list(existing_pos) + [r['Gene'] for r in pos_rows]
                print(f"\n{'─' * 64}")
                print(f'  EXTRACTING FEATURES for {len(fetched_genes)} genes')
                print(f"{'─' * 64}")
                for feat_fn, name in [(lambda g: get_mutation_features(g), 'Mutation'), (lambda g: get_mutation_context_domain_features(g), 'MutationContextDomain'), (lambda g: get_oncoplex_mutation_recurrence_features(g), 'ONCOPLEXMutationRecurrence'), (lambda g: get_intogen_driver_features(g), 'IntOGen'), (lambda g: get_cnv_features(g), 'CNV'), (lambda g: get_expression_features(g), 'Expression'), (lambda g: get_methylation_features(g), 'Methylation'), (lambda g: get_clinvar_features(g), 'ClinVar'), (lambda g: get_ppi_features(g, all_cgc), 'PPI'), (lambda g: get_pathway_features(g), 'Pathway')]:
                    try:
                        fdf = feat_fn(fetched_genes)
                        if fdf is None or fdf.empty:
                            continue
                        fdf = fdf.copy()
                        fdf['Gene'] = fdf['Gene'].astype(str).str.upper()
                        new_df = merge_feature_frame(new_df, fdf)
                        safe_to_csv(new_df, FEATURE_CACHE, index=False)
                        if name == 'PPI':
                            ppi_nonzero = int((new_df['PPI_degree'].fillna(0) > 0).sum())
                            cgc_nonzero = int((new_df['CGC_neighbor_count'].fillna(0) > 0).sum())
                            print(f'  PPI merged: {ppi_nonzero}/{len(new_df)} genes with PPI_degree, {cgc_nonzero}/{len(new_df)} with CGC_neighbor_count')
                    except Exception as e:
                        print(f'  {name} ERROR: {e} (skip)')
                safe_to_csv(new_df, FEATURE_CACHE, index=False)
            if ADD_ADVANCED_DRIVER_FEATURES:
                new_df = add_advanced_driver_features(new_df)
                new_df = move_label_to_end(new_df)
            else:
                new_df = drop_derived_proxy_features(new_df)
                new_df = move_label_to_end(new_df)
            lbl = new_df.pop('Label') if 'Label' in new_df.columns else pd.Series([0] * len(new_df))
            for col in existing_cols:
                if col not in new_df.columns:
                    new_df[col] = 0
            new_df = new_df.reindex(columns=existing_cols, fill_value=0).fillna(0)
            new_df['Label'] = lbl.values
            extended = pd.concat([base, new_df], ignore_index=True)
            extended = extended.drop_duplicates('Gene').reset_index(drop=True)
            extended['Label'] = pd.to_numeric(extended['Label'], errors='coerce').fillna(0).astype(int)
            extended = drop_derived_proxy_features(extended)
            extended = extended.fillna(0)
            if OVERSAMPLE_POSITIVES_TO_BALANCE:
                extended = oversample_positive_rows(extended, TARGET_NEG_POS_RATIO, RANDOM_STATE)
            extended = move_label_to_end(extended)
            saved_path = safe_to_csv(extended, OUTPUT_CSV, index=False)
            d1 = int((extended['Label'] == 1).sum())
            d0 = int((extended['Label'] == 0).sum())
            print('\n' + '=' * 64)
            print(f'DONE → {saved_path}')
            print(f"  Before: {len(base):4d} genes | {len(existing_pos)} pos | {(base['Label'] == 0).sum()} neg | {(base['Label'] == 0).sum() / len(existing_pos):.2f}:1")
            print(f'  After : {len(extended):4d} genes | {d1} pos | {d0} neg | {d0 / d1:.2f}:1')
            print(f'  Next  : python model_pipeline.py')
            print('=' * 64)
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_graph(forwarded_args):
    __file__ = str(ROOT / '01_code/_runtime/feature_graph.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import os
        import random
        import shutil
        import tempfile
        from collections import defaultdict, deque
        import numpy as np
        import pandas as pd
        from pathlib import Path
        REPO_ROOT = Path(__file__).resolve().parents[2]
        RAW_SOURCE_DIR = REPO_ROOT / '02_data' / 'raw_sources'
        INPUT_CSV = 'enhanced_sequence_dataset_balanced.csv'
        OUTPUT_CSV = 'enhanced_sequence_dataset_balanced_graph_features.csv'
        PPI_LINKS = str(RAW_SOURCE_DIR / '9606.protein.links.full.v12.0.txt')
        PPI_ALIASES = str(RAW_SOURCE_DIR / '9606.protein.aliases.v12.0.txt')
        CGC_FILE = str(RAW_SOURCE_DIR / 'Cosmic_CancerGeneCensus_Tsv_v103_GRCh37' / 'Cosmic_CancerGeneCensus_v103_GRCh37.tsv.gz')
        MIN_STRING_SCORE = 700
        BETWEENNESS_SAMPLES = 96
        RANDOM_STATE = 42

        def resolve_data_file(path):
            if os.path.isfile(path):
                return path
            if os.path.isdir(path):
                nested = os.path.join(path, os.path.basename(path))
                if os.path.isfile(nested):
                    return nested
                files = [os.path.join(path, f) for f in os.listdir(path)]
                files = [f for f in files if os.path.isfile(f)]
                if files:
                    return files[0]
            gz_path = path + '.gz'
            if os.path.isfile(gz_path):
                return gz_path
            return None

        def safe_read_csv(path, **kwargs):
            if str(path).endswith('.gz'):
                return pd.read_csv(path, **kwargs)
            tmp = os.path.join(tempfile.gettempdir(), '_graph_' + os.path.basename(path))
            try:
                shutil.copy2(path, tmp)
                return pd.read_csv(tmp, **kwargs)
            except PermissionError:
                return pd.read_csv(path, **kwargs)
            finally:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

        def load_cgc_genes():
            if not os.path.exists(CGC_FILE):
                print('CGC file not found; CGC neighborhood features will be zero.')
                return set()
            cgc = pd.read_csv(CGC_FILE, sep='\t', compression='gzip', low_memory=False)
            col = 'GENE_SYMBOL' if 'GENE_SYMBOL' in cgc.columns else cgc.columns[0]
            genes = set(cgc[col].dropna().astype(str).str.upper().str.strip())
            print(f'External CGC genes loaded: {len(genes)}')
            return genes

        def build_protein_to_gene_map(aliases_path, target_genes):
            print('Loading STRING aliases...')
            alias = safe_read_csv(aliases_path, sep='\t', low_memory=False)
            alias.columns = alias.columns.str.strip()
            prot_col = [c for c in alias.columns if 'protein' in c.lower()][0]
            alias_col = [c for c in alias.columns if 'alias' in c.lower()][0]
            src_col = [c for c in alias.columns if 'source' in c.lower()][0]
            alias[prot_col] = alias[prot_col].astype(str).str.strip()
            alias[alias_col] = alias[alias_col].astype(str).str.upper().str.strip()
            alias[src_col] = alias[src_col].astype(str)
            preferred = alias[alias[src_col].str.contains('HGNC|Gene_Name|BioMart_HUGO|Ensembl_HGNC', na=False, case=False)].copy()
            found = set(preferred[alias_col])
            missing = set(target_genes) - found
            if len(missing) > len(target_genes) * 0.3:
                fallback = alias[alias[alias_col].isin(missing)]
                combined = pd.concat([preferred, fallback], ignore_index=True)
            else:
                combined = preferred
            combined = combined.drop_duplicates([prot_col, alias_col]).sort_values(src_col)
            prot2gene = {}
            for row in combined[[prot_col, alias_col]].itertuples(index=False):
                if row[0] not in prot2gene:
                    prot2gene[row[0]] = row[1]
            print(f'Protein-to-gene aliases usable: {len(prot2gene):,}')
            return prot2gene

        def build_graph(links_path, prot2gene, target_genes):
            print('Loading STRING links and building high-confidence graph...')
            links = safe_read_csv(links_path, sep=' ', low_memory=False)
            links.columns = links.columns.str.strip()
            links = links[links['combined_score'] >= MIN_STRING_SCORE].copy()
            links['g1'] = links['protein1'].astype(str).str.strip().map(prot2gene)
            links['g2'] = links['protein2'].astype(str).str.strip().map(prot2gene)
            links = links.dropna(subset=['g1', 'g2'])
            links = links[links['g1'] != links['g2']]
            adjacency = defaultdict(set)
            weights = defaultdict(float)
            target_set = set(target_genes)
            target_edge_rows = []
            for g1, g2, score in links[['g1', 'g2', 'combined_score']].itertuples(index=False):
                score = float(score) / 1000.0
                edge = tuple(sorted((g1, g2)))
                if score > weights[edge]:
                    weights[edge] = score
                adjacency[g1].add(g2)
                adjacency[g2].add(g1)
                if g1 in target_set or g2 in target_set:
                    target_edge_rows.append((g1, g2, score))
            print(f'High-confidence mapped genes: {len(adjacency):,}')
            print(f'Edges touching dataset genes: {len(target_edge_rows):,}')
            return (adjacency, weights)

        def clustering_for_node(node, adjacency):
            neighbors = list(adjacency.get(node, ()))
            degree = len(neighbors)
            if degree < 2:
                return 0.0
            links_between_neighbors = 0
            neighbor_set = set(neighbors)
            for nbr in neighbors:
                links_between_neighbors += len(adjacency.get(nbr, set()) & neighbor_set)
            return links_between_neighbors / (degree * (degree - 1))

        def twohop_cgc_count(node, adjacency, cgc_genes):
            seen = set()
            direct = adjacency.get(node, set())
            for nbr in direct:
                seen.update(adjacency.get(nbr, set()))
            seen.discard(node)
            seen.difference_update(direct)
            return len(seen & cgc_genes)

        def approximate_betweenness_for_targets(adjacency, target_genes):
            target_set = set(target_genes)
            sub_adj = {gene: sorted(adjacency.get(gene, set()) & target_set) for gene in target_genes}
            active_nodes = [n for n, nbrs in sub_adj.items() if nbrs]
            if len(active_nodes) < 3:
                return {g: 0.0 for g in target_genes}
            rng = random.Random(RANDOM_STATE)
            sources = rng.sample(active_nodes, min(BETWEENNESS_SAMPLES, len(active_nodes)))
            bet = dict.fromkeys(target_genes, 0.0)
            for source in sources:
                stack = []
                pred = {w: [] for w in active_nodes}
                sigma = dict.fromkeys(active_nodes, 0.0)
                dist = dict.fromkeys(active_nodes, -1)
                sigma[source] = 1.0
                dist[source] = 0
                queue = deque([source])
                while queue:
                    v = queue.popleft()
                    stack.append(v)
                    for w in sub_adj.get(v, ()):
                        if dist.get(w, -1) < 0:
                            queue.append(w)
                            dist[w] = dist[v] + 1
                        if dist[w] == dist[v] + 1:
                            sigma[w] += sigma[v]
                            pred[w].append(v)
                delta = dict.fromkeys(active_nodes, 0.0)
                while stack:
                    w = stack.pop()
                    if sigma[w] > 0:
                        coeff = (1.0 + delta[w]) / sigma[w]
                        for v in pred[w]:
                            delta[v] += sigma[v] * coeff
                    if w != source:
                        bet[w] += delta[w]
            if len(sources) > 0:
                scale = len(active_nodes) / len(sources)
                norm = 1.0 / ((len(active_nodes) - 1) * (len(active_nodes) - 2))
                bet = {g: v * scale * norm for g, v in bet.items()}
            return bet

        def extract_graph_features(df, adjacency, weights, cgc_genes):
            genes = df['Gene'].astype(str).str.upper().str.strip().tolist()
            degree = {g: len(adjacency.get(g, set())) for g in genes}
            full_degrees = {g: len(nbrs) for g, nbrs in adjacency.items()}
            print('Computing approximate betweenness on dataset-gene subgraph...')
            betweenness = approximate_betweenness_for_targets(adjacency, genes)
            rows = []
            for gene in genes:
                neighbors = adjacency.get(gene, set())
                deg = degree[gene]
                weighted_degree = sum((weights.get(tuple(sorted((gene, nbr))), 0.0) for nbr in neighbors))
                neighbor_degrees = [full_degrees.get(nbr, 0) for nbr in neighbors]
                cgc_direct = len(neighbors & cgc_genes)
                rows.append({'Gene': gene, 'PPI_graph_degree': deg, 'PPI_graph_weighted_degree': weighted_degree, 'PPI_graph_avg_neighbor_degree': float(np.mean(neighbor_degrees)) if neighbor_degrees else 0.0, 'PPI_graph_max_neighbor_degree': int(max(neighbor_degrees)) if neighbor_degrees else 0, 'PPI_graph_clustering': clustering_for_node(gene, adjacency), 'PPI_graph_CGC_neighbor_count': cgc_direct, 'PPI_graph_CGC_neighbor_ratio': cgc_direct / deg if deg else 0.0, 'PPI_graph_twohop_CGC_count': twohop_cgc_count(gene, adjacency, cgc_genes), 'PPI_graph_betweenness_approx': betweenness.get(gene, 0.0)})
            return pd.DataFrame(rows)

        def main():
            if not os.path.exists(INPUT_CSV):
                raise FileNotFoundError(f'Input CSV not found: {INPUT_CSV}')
            links_path = resolve_data_file(PPI_LINKS)
            aliases_path = resolve_data_file(PPI_ALIASES)
            if not links_path or not aliases_path:
                raise FileNotFoundError('STRING PPI links/aliases files not found.')
            df = pd.read_csv(INPUT_CSV)
            df.columns = df.columns.str.strip()
            df['Gene'] = df['Gene'].astype(str).str.upper().str.strip()
            target_genes = sorted(df['Gene'].dropna().unique())
            print(f'Input rows: {len(df):,}')
            print(f'Unique genes: {len(target_genes):,}')
            print(f'Links file: {links_path}')
            print(f'Aliases file: {aliases_path}')
            cgc_genes = load_cgc_genes()
            prot2gene = build_protein_to_gene_map(aliases_path, target_genes)
            adjacency, weights = build_graph(links_path, prot2gene, target_genes)
            graph_features = extract_graph_features(df, adjacency, weights, cgc_genes)
            graph_cols = [c for c in graph_features.columns if c != 'Gene']
            out = df.drop(columns=graph_cols, errors='ignore').merge(graph_features, on='Gene', how='left')
            out[graph_cols] = out[graph_cols].fillna(0)
            out.to_csv(OUTPUT_CSV, index=False)
            print(f'Graph feature columns added: {len(graph_cols)}')
            print(f'Output shape: {out.shape}')
            print(f'Saved: {OUTPUT_CSV}')
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_engineered(forwarded_args):
    __file__ = str(ROOT / '01_code/_runtime/feature_engineered.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import numpy as np
        import pandas as pd
        INPUT_CSV = 'enhanced_sequence_dataset_balanced_graph_features.csv'
        OUTPUT_CSV = 'enhanced_sequence_dataset_balanced_graph_engineered.csv'

        def col(df, name, default=0.0):
            if name in df.columns:
                return pd.to_numeric(df[name], errors='coerce').fillna(default)
            return pd.Series(default, index=df.index, dtype=float)

        def add_feature(df, name, values):
            df[name] = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

        def main():
            df = pd.read_csv(INPUT_CSV)
            df.columns = df.columns.str.strip()
            df['Gene'] = df['Gene'].astype(str).str.upper().str.strip()
            out = df.copy()
            count_cols = ['SNV_Count', 'Missense', 'Nonsense', 'Frameshift', 'Silent', 'Splice_Site', 'Patients_Affected', 'ClinVar_Total_Variants', 'ClinVar_Pathogenic_Count', 'ClinVar_Benign_Count', 'ClinVar_Uncertain_Count', 'ClinVar_Reviewed_Pathogenic', 'ClinVar_SNV_Count', 'ClinVar_Indel_Count', 'ClinVar_Structural_Count', 'PPI_degree', 'CGC_neighbor_count', 'pathway_count_all', 'PPI_graph_degree', 'PPI_graph_weighted_degree', 'PPI_graph_avg_neighbor_degree', 'PPI_graph_max_neighbor_degree', 'PPI_graph_CGC_neighbor_count', 'PPI_graph_twohop_CGC_count', 'Prom_Meth_Probe_Count', 'IntOGen_Driver_Cohort_Count', 'IntOGen_Driver_CancerType_Count', 'IntOGen_Total_Mutations', 'IntOGen_Max_Cohort_Mutations', 'IntOGen_Total_Mutated_Samples', 'IntOGen_Max_Cohort_Samples', 'IntOGen_Unfiltered_Cohort_Count', 'IntOGen_Unfiltered_CancerType_Count', 'IntOGen_CancerMine_NumPapers', 'Hotspot_Max_Same_Position', 'Hotspot_Recurrent_Position_Count', 'Domain_Mutation_Count', 'Domain_Unique_Count', 'SIFT_Deleterious_Count', 'PolyPhen_Damaging_Count']
            for c in count_cols:
                if c in out.columns:
                    x = col(out, c).clip(lower=0)
                    add_feature(out, f'FE_log1p_{c}', np.log1p(x))
                    add_feature(out, f'FE_sqrt_{c}', np.sqrt(x))
            snv = col(out, 'SNV_Count')
            patients = col(out, 'Patients_Affected')
            non_silent = col(out, 'Missense') + col(out, 'Nonsense') + col(out, 'Frameshift') + col(out, 'Splice_Site')
            trunc = col(out, 'Nonsense') + col(out, 'Frameshift') + col(out, 'Splice_Site')
            add_feature(out, 'FE_mutations_per_patient', snv / (patients + 1))
            add_feature(out, 'FE_non_silent_count', non_silent)
            add_feature(out, 'FE_non_silent_per_patient', non_silent / (patients + 1))
            add_feature(out, 'FE_truncating_count', trunc)
            add_feature(out, 'FE_truncating_per_patient', trunc / (patients + 1))
            add_feature(out, 'FE_truncating_to_missense', trunc / (col(out, 'Missense') + 1))
            add_feature(out, 'FE_high_impact_mutation_load', col(out, 'High_Impact_Ratio') * np.log1p(snv.clip(lower=0)))
            add_feature(out, 'FE_pathogenicity_mutation_load', col(out, 'Pathogenicity_Score') * np.log1p(non_silent.clip(lower=0)))
            add_feature(out, 'FE_hotspot_mutation_load', col(out, 'Hotspot_Fraction') * np.log1p(snv.clip(lower=0)))
            add_feature(out, 'FE_domain_deleterious_load', col(out, 'Domain_Mutation_Ratio') * col(out, 'Deleterious_Impact_Ratio'))
            ppi = col(out, 'PPI_degree')
            cgc = col(out, 'CGC_neighbor_count')
            pathway = col(out, 'pathway_count_all')
            gdeg = col(out, 'PPI_graph_degree')
            gwdeg = col(out, 'PPI_graph_weighted_degree')
            gcgc = col(out, 'PPI_graph_CGC_neighbor_count')
            twohop = col(out, 'PPI_graph_twohop_CGC_count')
            bet = col(out, 'PPI_graph_betweenness_approx')
            clustering = col(out, 'PPI_graph_clustering')
            add_feature(out, 'FE_network_total_signal', np.log1p(ppi + gdeg + gwdeg + cgc + gcgc + pathway))
            add_feature(out, 'FE_graph_cgc_density', gcgc / (gdeg + 1))
            add_feature(out, 'FE_graph_twohop_density', twohop / (gdeg + 1))
            add_feature(out, 'FE_graph_broker_score', np.log1p(gdeg.clip(lower=0)) * bet)
            add_feature(out, 'FE_graph_module_score', np.log1p(gcgc + twohop) * (1 - clustering.clip(0, 1)))
            add_feature(out, 'FE_pathway_graph_load', np.log1p(pathway.clip(lower=0)) * np.log1p(gdeg.clip(lower=0)))
            add_feature(out, 'FE_cgc_pathway_bridge', np.log1p(cgc + gcgc) * np.log1p(pathway.clip(lower=0)))
            add_feature(out, 'FE_weighted_degree_density', gwdeg / (gdeg + 1))
            clin_total = col(out, 'ClinVar_Total_Variants')
            clin_path = col(out, 'ClinVar_Pathogenic_Count')
            clin_benign = col(out, 'ClinVar_Benign_Count')
            add_feature(out, 'FE_clinvar_pathogenic_burden', clin_path / (clin_total + 1))
            add_feature(out, 'FE_clinvar_pathogenic_net', (clin_path - clin_benign) / (clin_total + 1))
            add_feature(out, 'FE_clinvar_reviewed_pathogenic_burden', col(out, 'ClinVar_Reviewed_Pathogenic') / (clin_total + 1))
            add_feature(out, 'FE_clinvar_graph_interaction', np.log1p(clin_path.clip(lower=0)) * np.log1p(gcgc.clip(lower=0)))
            gain = col(out, 'CNV_Gain_Ratio')
            loss = col(out, 'CNV_Loss_Ratio')
            amp = col(out, 'CNV_Amplification_Ratio')
            dele = col(out, 'CNV_Deletion_Ratio')
            add_feature(out, 'FE_cnv_gain_loss_delta', gain - loss)
            add_feature(out, 'FE_cnv_amp_del_delta', amp - dele)
            add_feature(out, 'FE_cnv_instability', gain + loss + amp + dele)
            add_feature(out, 'FE_cnv_expression_load', col(out, 'CNV_Mean').abs() * col(out, 'Expr_CV').abs())
            add_feature(out, 'FE_methylation_delta', col(out, 'Meth_Hyper_Ratio') - col(out, 'Meth_Hypo_Ratio'))
            add_feature(out, 'FE_promoter_methylation_confidence', col(out, 'Prom_Meth_Mean') * np.log1p(col(out, 'Prom_Meth_Probe_Count').clip(lower=0)))
            add_feature(out, 'FE_expr_meth_interaction', col(out, 'Expr_CV') * (col(out, 'Meth_Hyper_Ratio') - col(out, 'Meth_Hypo_Ratio')))
            q = col(out, 'IntOGen_Min_QValue')
            add_feature(out, 'FE_intogen_q_score', -np.log10(q.clip(lower=1e-300) + 1e-300))
            add_feature(out, 'FE_intogen_driver_evidence', np.log1p(col(out, 'IntOGen_Driver_Cohort_Count') + col(out, 'IntOGen_Driver_CancerType_Count')) * (col(out, 'IntOGen_Is_Compendium_Driver') + 1))
            add_feature(out, 'FE_intogen_sample_mutation_density', col(out, 'IntOGen_Total_Mutations') / (col(out, 'IntOGen_Total_Mutated_Samples') + 1))
            add_feature(out, 'FE_gc_entropy_balance', col(out, 'GC_Content') * col(out, 'Entropy_2mer'))
            add_feature(out, 'FE_cpg_promoter_meth', col(out, 'CpG_Density') * col(out, 'Prom_Meth_Mean'))
            before_cols = len(df.columns)
            after_cols = len(out.columns)
            out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
            out.to_csv(OUTPUT_CSV, index=False)
            print(f'Input shape: {df.shape}')
            print(f'Output shape: {out.shape}')
            print(f'Engineered columns added: {after_cols - before_cols}')
            print(f'Saved: {OUTPUT_CSV}')
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['all', 'sequences', 'graph', 'engineered'])
    arguments = _sys.argv[1:]
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser.print_help()
        return
    stage = parser.parse_args(arguments[:1]).stage
    forwarded = arguments[1:]
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if forwarded:
        if forwarded in (['-h'], ['--help']):
            parser.print_help()
            return
        parser.error('feature stages do not accept extra arguments')
    if stage == 'all':
        for name in ['sequences', 'graph', 'engineered']:
            {'sequences': _run_sequences, 'graph': _run_graph, 'engineered': _run_engineered}[name]([])
        return
    {'sequences': _run_sequences, 'graph': _run_graph, 'engineered': _run_engineered}[stage]([])

if __name__ == '__main__':
    main()
