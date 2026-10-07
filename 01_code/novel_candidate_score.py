from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _run_candidates(forwarded_args):
    __file__ = str(ROOT / '01_code/_runtime/candidate_discovery.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import json
        import sys
        import zipfile
        import os
        from pathlib import Path
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        WORK_DIR = Path(__file__).resolve().parents[2]
        DATA_SOURCES = WORK_DIR / '02_data' / 'raw_sources'
        EXPERIMENT = WORK_DIR / os.environ.get('MOSVE_EXPERIMENT_NAME', '04_strict_verified_4420_experiment')
        DATA = EXPERIMENT / 'data'
        PRIMARY_RESULTS = EXPERIMENT / 'results'
        OUT = EXPERIMENT / 'resource_filtered_candidate_validation'
        OUT.mkdir(parents=True, exist_ok=True)
        MATRIX = DATA / 'strict_verified_4420_feature_matrix.csv'
        SPLITS = DATA / 'strict_verified_4420_split_membership.csv'
        LABEL_AUDIT = DATA / 'strict_verified_4420_label_audit.csv'
        SEEDS = (42, 7, 101, 2024)
        FOLDS = 5
        MI_K = 200
        HIGH_SCORE = 0.8

        def source_derived_feature(name: str) -> bool:
            low = name.lower()
            if name == 'k3_CGC':
                return False
            if 'intogen' in low:
                return True
            if 'cgc_neighbor' in low or 'twohop_cgc' in low or 'cgc_pathway' in low:
                return True
            return name in {'CGC_neighbor_count', 'PPI_graph_CGC_neighbor_count', 'PPI_graph_CGC_neighbor_ratio', 'FE_network_total_signal', 'FE_graph_module_score', 'FE_graph_twohop_density', 'FE_clinvar_graph_interaction'}

        def pipeline(model, scale: bool=False) -> Pipeline:
            steps = [('imputer', SimpleImputer(strategy='median'))]
            if scale:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', model))
            return Pipeline(steps)

        def mosve(seed: int) -> VotingClassifier:
            estimators = [('ExtraTrees', pipeline(ExtraTreesClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=0.5, class_weight='balanced', random_state=seed, n_jobs=-1))), ('HistGradientBoosting', pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1, random_state=seed))), ('LogisticRegression', pipeline(LogisticRegression(C=0.06951927961775606, solver='liblinear', penalty='l2', max_iter=7000, class_weight='balanced', random_state=seed), scale=True)), ('LightGBM', pipeline(LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=0.0, random_state=seed, n_jobs=-1, verbose=-1)))]
            return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)

        def select_features(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> list[str]:
            values = SimpleImputer(strategy='median').fit_transform(X_train)
            scaled = MinMaxScaler().fit_transform(values)
            mi = mutual_info_classif(scaled, y_train, random_state=seed)
            ranked = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
            return ranked.head(min(MI_K, len(ranked))).index.tolist()

        def run_repeated_oof(X: pd.DataFrame, y: pd.Series, genes: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
            all_scores: list[pd.DataFrame] = []
            metric_rows: list[dict] = []
            for seed in SEEDS:
                print(f'Resource-filtered ranking seed {seed}', flush=True)
                splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
                scores = np.zeros(len(X), dtype=float)
                for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
                    print(f'  fold {fold}/{FOLDS}', flush=True)
                    selected = select_features(X.iloc[train_idx], y.iloc[train_idx], seed + fold)
                    model = mosve(seed + fold)
                    model.fit(X.iloc[train_idx][selected], y.iloc[train_idx])
                    scores[test_idx] = model.predict_proba(X.iloc[test_idx][selected])[:, 1]
                all_scores.append(pd.DataFrame({'Gene': genes.values, f'OOF_Score_Seed{seed}': scores}))
                metric_rows.append({'Seed': seed, 'ROC_AUC': roc_auc_score(y, scores), 'AUPRC': average_precision_score(y, scores)})
            merged = all_scores[0]
            for frame in all_scores[1:]:
                merged = merged.merge(frame, on='Gene', how='inner')
            score_cols = [c for c in merged if c.startswith('OOF_Score_')]
            merged['Mean_OOF_Score'] = merged[score_cols].mean(axis=1)
            merged['SD_OOF_Score'] = merged[score_cols].std(axis=1)
            merged['High_Score_Repetitions'] = merged[score_cols].ge(HIGH_SCORE).sum(axis=1)
            merged['Stable_High_Score'] = merged['High_Score_Repetitions'].ge(3).astype(int)
            return (merged, pd.DataFrame(metric_rows))

        def read_symbols(path: Path, column: str, sep: str='\t') -> set[str]:
            return set(pd.read_csv(path, sep=sep)[column].dropna().astype(str).str.strip().str.upper())

        def external_evidence(candidates: pd.DataFrame) -> pd.DataFrame:
            root = DATA_SOURCES
            intogen_zip = root / 'ONCOPLEX-main' / 'ONCOPLEX-main' / 'Data' / 'IntOGEN_Drivers.zip'
            with zipfile.ZipFile(intogen_zip) as archive:
                intogen = set().union(*[read_symbols_from_handle(archive.open(name), 'Symbol') for name in archive.namelist() if name.endswith('.tsv')])
            civic = read_symbols(DATA_SOURCES / 'CIViC' / 'nightly-GeneSummaries.tsv', 'name')
            pcawg_root = WORK_DIR / '05_external_validation' / 'PCAWG_ICGC'
            events = pd.read_csv(pcawg_root / 'PCAWG_driver_events_ICGC.tsv.gz', sep='\t')
            events['Gene'] = events['gene'].astype(str).str.upper()
            events = events.groupby('Gene', as_index=False).agg(PCAWG_Driver_Events=('Gene', 'size'), PCAWG_Driver_Samples=('sample_id', 'nunique'), PCAWG_Driver_Types=('ttype', 'nunique'))
            compendium = pd.read_excel(pcawg_root / 'PCAWG_driver_compendium.xlsx', sheet_name='TableS1_compendium_mutational_drivers_Jan12_2018')
            compendium['Gene'] = compendium['Gene'].astype(str).str.upper()
            compendium = compendium.groupby('Gene', as_index=False).agg(PCAWG_Compendium_Entries=('Gene', 'size'), PCAWG_Compendium_Tissues=('Tissue', 'nunique'))
            out = candidates.copy()
            out['IntOGen_External'] = out['Gene'].isin(intogen).astype(int)
            out['CIViC_Curated'] = out['Gene'].isin(civic).astype(int)
            out = out.merge(events, on='Gene', how='left').merge(compendium, on='Gene', how='left')
            for col in ['PCAWG_Driver_Events', 'PCAWG_Driver_Samples', 'PCAWG_Driver_Types', 'PCAWG_Compendium_Entries', 'PCAWG_Compendium_Tissues']:
                out[col] = out[col].fillna(0).astype(int)
            out['Independent_Resource_Support'] = out['IntOGen_External'] + out['CIViC_Curated'] + out['PCAWG_Driver_Events'].gt(0).astype(int) + out['PCAWG_Compendium_Entries'].gt(0).astype(int)
            return out

        def read_symbols_from_handle(handle, column: str) -> set[str]:
            return set(pd.read_csv(handle, sep='\t')[column].dropna().astype(str).str.strip().str.upper())

        def main() -> None:
            matrix = pd.read_csv(MATRIX)
            matrix['Gene'] = matrix['Gene'].astype(str).str.upper()
            membership = pd.read_csv(SPLITS)
            membership['Gene'] = membership['Gene'].astype(str).str.upper()
            audit = pd.read_csv(LABEL_AUDIT)
            audit['Gene'] = audit['Gene'].astype(str).str.upper()
            source_cols = ['CGC', 'OncoKB', 'OnGene', 'NCG']
            missing = [col for col in source_cols if col not in audit]
            if missing:
                raise ValueError(f'Missing label-audit columns: {missing}')
            excluded = [c for c in matrix.columns if source_derived_feature(c)]
            features = [c for c in matrix.columns if c not in {'Gene', 'Label'} and c not in excluded]
            if len(features) != 249:
                raise ValueError(f'Expected 249 resource-filtered features, found {len(features)}')
            pd.DataFrame({'Excluded_Feature': excluded}).to_csv(OUT / 'excluded_resource_features.csv', index=False)
            pd.DataFrame({'Retained_Feature': features}).to_csv(OUT / 'retained_resource_filtered_features.csv', index=False)
            development_genes = set(membership.loc[membership['Split'].eq('development'), 'Gene'])
            development = matrix[matrix['Gene'].isin(development_genes)].copy()
            scores, metrics = run_repeated_oof(development[features], development['Label'].astype(int), development['Gene'])
            metrics.to_csv(OUT / 'repeated_development_oof_metrics.csv', index=False)
            candidates = scores.merge(development[['Gene', 'Label']], on='Gene', how='left').merge(audit[['Gene', *source_cols]], on='Gene', how='left')
            candidates['Any_Label_Source'] = candidates[source_cols].sum(axis=1).gt(0).astype(int)
            candidates = candidates.merge(development[['Gene', 'SNV_Count', 'Patients_Affected', 'Missense', 'Nonsense', 'Frameshift', 'Pathogenicity_Score', 'High_Impact_Ratio', 'Mutation_Impact_Score', 'Expr_Mean', 'CNV_Mean', 'Meth_Mean', 'PPI_degree', 'pathway_count_all']], on='Gene', how='left')
            candidates = external_evidence(candidates)
            candidates['Candidate_Class'] = 'Not selected'
            eligible = candidates['Label'].eq(0) & candidates['Any_Label_Source'].eq(0)
            candidates.loc[eligible & candidates['Stable_High_Score'].eq(1), 'Candidate_Class'] = 'Stable computational candidate'
            candidates.loc[eligible & candidates['Stable_High_Score'].eq(1) & candidates['Independent_Resource_Support'].gt(0), 'Candidate_Class'] = 'Stable candidate with independent support'
            candidates.loc[eligible & candidates['Stable_High_Score'].eq(1) & candidates['Independent_Resource_Support'].eq(0), 'Candidate_Class'] = 'Stable database-absent candidate'
            candidates = candidates.sort_values(['Candidate_Class', 'Mean_OOF_Score', 'SD_OOF_Score'], ascending=[True, False, True])
            candidates.to_csv(OUT / 'all_development_gene_candidate_scores.csv', index=False)
            stable = candidates[candidates['Candidate_Class'].ne('Not selected')].copy()
            stable.to_csv(OUT / 'stable_resource_filtered_candidates.csv', index=False)
            externally_supported = stable[stable['Independent_Resource_Support'].gt(0)].copy()
            externally_supported.to_csv(OUT / 'stable_candidates_with_independent_support.csv', index=False)
            database_absent = stable[stable['Independent_Resource_Support'].eq(0)].copy()
            database_absent.to_csv(OUT / 'stable_database_absent_candidates.csv', index=False)
            primary = database_absent.sort_values(['Mean_OOF_Score', 'SD_OOF_Score'], ascending=[False, True]).head(5).copy()
            primary.to_csv(OUT / 'primary_database_absent_candidate_shortlist.csv', index=False)
            summary = {'cohort': EXPERIMENT.name, 'development_genes': int(len(development)), 'resource_filtered_features': int(len(features)), 'excluded_resource_features': int(len(excluded)), 'seeds': list(SEEDS), 'folds_per_seed': FOLDS, 'high_score_cutoff': HIGH_SCORE, 'stable_candidates': int(len(stable)), 'stable_candidates_with_independent_support': int(len(externally_supported)), 'stable_database_absent_candidates': int(len(database_absent)), 'primary_shortlist_size': int(len(primary)), 'ensemble': 'Extra Trees + HistGradientBoosting + Logistic Regression + LightGBM', 'mean_repeated_oof_roc_auc': round(float(metrics['ROC_AUC'].mean()), 4), 'mean_repeated_oof_auprc': round(float(metrics['AUPRC'].mean()), 4)}
            (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(json.dumps(summary, indent=2))
        if __name__ == '__main__':
            try:
                main()
            except Exception as error:
                print(f'ERROR: {error}', file=sys.stderr)
                raise
    finally:
        _sys.argv = _runner_previous_argv


def main():
    parser = argparse.ArgumentParser(description='Calculate MOSVE novel-candidate scores.')
    parser.parse_args()
    _run_candidates([])

if __name__ == '__main__':
    main()
