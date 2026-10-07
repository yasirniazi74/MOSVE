from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def validate_release_matches(cohort, release_path):
    import pandas as pd
    release = pd.read_csv(release_path, usecols=["Gene", "Label"])
    release["Gene"] = release["Gene"].astype(str).str.strip().str.upper()
    release["Label"] = pd.to_numeric(release["Label"], errors="raise").astype(int)
    cohort = cohort[["Gene", "Label"]].copy()
    cohort["Gene"] = cohort["Gene"].astype(str).str.strip().str.upper()
    cohort["Label"] = pd.to_numeric(cohort["Label"], errors="raise").astype(int)
    if release["Gene"].duplicated().any():
        raise ValueError(f"Bundled gene-label release has duplicate symbols: {release_path}")
    expected = release.set_index("Gene")["Label"].sort_index()
    observed = cohort.set_index("Gene")["Label"].sort_index()
    if not expected.equals(observed):
        shared = expected.index.intersection(observed.index)
        mismatches = (expected.loc[shared].to_numpy() != observed.loc[shared].to_numpy()).sum()
        raise ValueError(f"Reconstructed cohort differs from bundled release: missing={len(expected.index.difference(observed.index))}, extra={len(observed.index.difference(expected.index))}, label_mismatches={mismatches}.")

def _run_train(forwarded_args):
    __file__ = str(ROOT / '01_code/_runtime/model_training.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import json
        import os
        import warnings
        from pathlib import Path
        os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))
        warnings.filterwarnings('ignore')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
        from sklearn.model_selection import StratifiedKFold, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        WORK_DIR = Path(__file__).resolve().parents[2]
        DATA_SOURCES = WORK_DIR / '02_data' / 'raw_sources'
        EXPERIMENT_DIR = WORK_DIR / os.environ.get('MOSVE_EXPERIMENT_NAME', '04_strict_verified_4420_experiment')
        DATA_DIR = EXPERIMENT_DIR / 'data'
        RESULTS_DIR = EXPERIMENT_DIR / 'results'
        RAW_MATRIX = WORK_DIR / '02_data' / 'final_model_inputs' / 'enhanced_sequence_dataset_balanced_graph_engineered.csv'
        LABEL_LEDGER = WORK_DIR / '02_data' / 'label_sources' / 'all_gene_labels_combined.csv'
        TARGET_COL = 'Label'
        ID_COL = 'Gene'
        SEED = int(os.environ.get('MOSVE_CV_SEED', '42'))
        SPLIT_SEED = int(os.environ.get('MOSVE_SPLIT_SEED', '42'))
        TEST_SIZE = 0.2
        INNER_VALIDATION_SIZE = 0.2
        N_FOLDS = 10
        MI_K = 200
        EXCLUDE_INTOGEN_FEATURES = os.environ.get('MOSVE_EXCLUDE_INTOGEN_FEATURES', '0') == '1'

        def clean_symbols(values) -> set[str]:
            return {str(value).strip().upper() for value in values if pd.notna(value) and str(value).strip()}

        def validate_bundled_gene_labels(cohort: pd.DataFrame) -> None:
            release_path = WORK_DIR / '02_data' / 'labels' / 'gene_labels.csv'
            release = pd.read_csv(release_path, usecols=[ID_COL, TARGET_COL])
            release[ID_COL] = release[ID_COL].astype(str).str.strip().str.upper()
            release[TARGET_COL] = pd.to_numeric(release[TARGET_COL], errors='raise').astype(int)
            if release[ID_COL].duplicated().any():
                raise ValueError(f'Bundled gene-label release has duplicate symbols: {release_path}')
            expected = release.set_index(ID_COL)[TARGET_COL].sort_index()
            observed = cohort.set_index(ID_COL)[TARGET_COL].astype(int).sort_index()
            if not expected.equals(observed):
                missing = expected.index.difference(observed.index)
                extra = observed.index.difference(expected.index)
                mismatched = expected.index.intersection(observed.index)[expected.loc[expected.index.intersection(observed.index)].to_numpy() != observed.loc[expected.index.intersection(observed.index)].to_numpy()]
                raise ValueError(f'Reconstructed cohort differs from the bundled 4,420-gene release: missing={len(missing)}, extra={len(extra)}, label_mismatches={len(mismatched)}.')

        def approved_sources() -> dict[str, set[str]]:
            return {'CGC': clean_symbols(pd.read_csv(DATA_SOURCES / 'Cosmic_CancerGeneCensus_Tsv_v103_GRCh37' / 'Cosmic_CancerGeneCensus_v103_GRCh37.tsv.gz', sep='\t', compression='gzip', low_memory=False)['GENE_SYMBOL']), 'OncoKB': clean_symbols(pd.read_csv(DATA_SOURCES / 'ONCOPLEX-main' / 'Data' / 'Additional gene sets' / 'oncoKB_genes.tsv', sep='\t', low_memory=False)['Hugo Symbol']), 'OnGene': clean_symbols(pd.read_csv(DATA_SOURCES / 'ONCOPLEX-main' / 'Data' / 'Additional gene sets' / 'ongene_human.txt', sep='\t', low_memory=False)['OncogeneName']), 'NCG': clean_symbols(pd.read_csv(WORK_DIR / '02_data' / 'database_tables_small' / 'NCG_cancerdrivers_annotation_supporting_evidence.tsv', sep='\t', low_memory=False)['symbol'])}

        def build_strict_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
            raw = pd.read_csv(RAW_MATRIX)
            raw.columns = raw.columns.str.strip()
            raw[ID_COL] = raw[ID_COL].astype(str).str.strip().str.upper()
            raw[TARGET_COL] = pd.to_numeric(raw[TARGET_COL], errors='raise').astype(int)
            if raw[ID_COL].duplicated().any():
                raise ValueError('The raw matrix contains duplicate gene symbols.')
            ledger = pd.read_csv(LABEL_LEDGER, usecols=['Gene', 'Positive_Source', 'Conflict_NegativeWins'])
            ledger['Gene'] = ledger['Gene'].astype(str).str.strip().str.upper()
            ledger['Conflict_NegativeWins'] = pd.to_numeric(ledger['Conflict_NegativeWins'], errors='coerce').fillna(0).astype(int)
            audit = raw[[ID_COL, TARGET_COL]].rename(columns={TARGET_COL: 'Original_Label'}).merge(ledger, on=ID_COL, how='left')
            sources = approved_sources()
            approved_union = set().union(*sources.values())
            for source_name, genes in sources.items():
                audit[source_name] = audit[ID_COL].isin(genes).astype(int)
            audit['Approved_Source_Membership'] = audit.apply(lambda row: '|'.join((name for name in sources if row[name] == 1)), axis=1)
            audit['Approved_Driver_Evidence'] = audit[ID_COL].isin(approved_union)
            audit['Working_Label'] = audit['Original_Label']
            audit['Action'] = 'retained'
            relabel_verified = audit['Working_Label'].eq(0) & audit['Conflict_NegativeWins'].eq(1) & audit['Approved_Driver_Evidence']
            if int(relabel_verified.sum()) != 194:
                raise RuntimeError(f'Expected 194 verified conflicts; found {int(relabel_verified.sum())}.')
            audit.loc[relabel_verified, 'Working_Label'] = 1
            audit.loc[relabel_verified, 'Action'] = 'relabelled_verified_negative_conflict_to_positive'
            cancer_mine_dependent = audit['Working_Label'].eq(1) & audit['Positive_Source'].fillna('').str.contains('CancerMine', regex=False) & ~audit['Approved_Driver_Evidence']
            if int(cancer_mine_dependent.sum()) != 596:
                raise RuntimeError(f'Expected 596 CancerMine-dependent positives; found {int(cancer_mine_dependent.sum())}.')
            audit.loc[cancer_mine_dependent, 'Action'] = 'excluded_cancermine_dependent_positive'
            unsupported_positive = audit['Working_Label'].eq(1) & ~audit['Approved_Driver_Evidence'] & ~cancer_mine_dependent
            if int(unsupported_positive.sum()) != 143:
                raise RuntimeError(f'Expected 143 unsupported positives; found {int(unsupported_positive.sum())}.')
            audit.loc[unsupported_positive, 'Action'] = 'excluded_positive_without_approved_evidence'
            approved_negative = audit['Working_Label'].eq(0) & audit['Approved_Driver_Evidence']
            if int(approved_negative.sum()) != 259:
                raise RuntimeError(f'Expected 259 approved-source negatives; found {int(approved_negative.sum())}.')
            audit.loc[approved_negative, 'Action'] = 'excluded_negative_with_approved_driver_evidence'
            keep_mask = audit['Action'].isin(['retained', 'relabelled_verified_negative_conflict_to_positive'])
            retained_genes = set(audit.loc[keep_mask, ID_COL])
            cohort = raw[raw[ID_COL].isin(retained_genes)].copy()
            new_labels = audit.set_index(ID_COL).loc[cohort[ID_COL], 'Working_Label'].astype(int).to_numpy()
            cohort[TARGET_COL] = new_labels
            audit['Final_In_Cohort'] = audit[ID_COL].isin(retained_genes).astype(int)
            audit['Final_Label'] = np.where(audit['Final_In_Cohort'].eq(1), audit['Working_Label'], np.nan)
            expected = {'rows': 4420, 'positive': 1689, 'negative': 2731}
            observed = {'rows': int(len(cohort)), 'positive': int(cohort[TARGET_COL].eq(1).sum()), 'negative': int(cohort[TARGET_COL].eq(0).sum())}
            if observed != expected:
                raise RuntimeError(f'Strict cohort counts mismatch: expected {expected}, observed {observed}.')
            if set(cohort.loc[cohort[TARGET_COL].eq(1), ID_COL]) - approved_union:
                raise RuntimeError('Strict cohort still contains a positive without approved evidence.')
            if set(cohort.loc[cohort[TARGET_COL].eq(0), ID_COL]) & approved_union:
                raise RuntimeError('Strict cohort still contains a negative with approved driver evidence.')
            profile = {'raw_matrix': str(RAW_MATRIX), 'approved_sources': {name: len(genes) for name, genes in sources.items()}, 'relabelled_verified_conflicts': int(relabel_verified.sum()), 'excluded_cancermine_dependent_positive': int(cancer_mine_dependent.sum()), 'excluded_unsupported_positive': int(unsupported_positive.sum()), 'excluded_negative_with_driver_evidence': int(approved_negative.sum()), **observed, 'features': int(cohort.drop(columns=[ID_COL, TARGET_COL], errors='ignore').shape[1])}
            return (cohort, audit, profile)

        def make_pipeline(model, scale: bool=False) -> Pipeline:
            steps = [('imputer', SimpleImputer(strategy='median'))]
            if scale:
                steps.append(('scaler', StandardScaler()))
            steps.append(('model', model))
            return Pipeline(steps)
        MEMBER_DISPLAY = {'ExtraTrees': 'Extra Trees', 'HistGradientBoosting': 'HistGradientBoosting', 'LogisticRegression': 'Logistic Regression', 'LightGBM': 'LightGBM'}
        MOSVE_DISPLAY = 'MOSVE: HGB + ET + LR + LGBM'
        LEAVE_ONE_OUT = {'Without Extra Trees': ['HistGradientBoosting', 'LogisticRegression', 'LightGBM'], 'Without HistGradientBoosting': ['ExtraTrees', 'LogisticRegression', 'LightGBM'], 'Without Logistic Regression': ['ExtraTrees', 'HistGradientBoosting', 'LightGBM'], 'Without LightGBM': ['ExtraTrees', 'HistGradientBoosting', 'LogisticRegression']}

        def build_mosve() -> VotingClassifier:
            estimators = [('ExtraTrees', make_pipeline(ExtraTreesClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=0.5, max_depth=None, class_weight='balanced', random_state=SEED, n_jobs=1))), ('HistGradientBoosting', make_pipeline(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED))), ('LogisticRegression', make_pipeline(LogisticRegression(C=0.06951927961775606, solver='liblinear', penalty='l2', max_iter=7000, class_weight='balanced', random_state=SEED), scale=True)), ('LightGBM', make_pipeline(LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=0.0, random_state=SEED, n_jobs=1, verbose=-1)))]
            return VotingClassifier(estimators=estimators, voting='soft', n_jobs=1)

        def member_probabilities(model: VotingClassifier, X: pd.DataFrame) -> dict[str, np.ndarray]:
            return {name: estimator.predict_proba(X)[:, 1] for name, estimator in model.named_estimators_.items()}

        def average_scores(score_map: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
            return np.mean(np.vstack([score_map[name] for name in names]), axis=0)

        def select_mi_features(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[list[str], pd.DataFrame]:
            imputed = SimpleImputer(strategy='median').fit_transform(X_train)
            scaled = MinMaxScaler().fit_transform(imputed)
            scores = mutual_info_classif(scaled, y_train, random_state=SEED)
            ranking = pd.DataFrame({'Feature': X_train.columns, 'MI': scores}).sort_values('MI', ascending=False)
            return (ranking.head(min(MI_K, X_train.shape[1]))['Feature'].tolist(), ranking)

        def tune_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
            candidates = np.linspace(0.05, 0.95, 181)
            f1s = [f1_score(y_true, scores >= threshold, zero_division=0) for threshold in candidates]
            return float(candidates[int(np.argmax(f1s))])

        def metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict:
            y_true = np.asarray(y_true)
            pred = (scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            return {'Threshold': round(float(threshold), 3), 'ROC_AUC': round(float(roc_auc_score(y_true, scores)), 4), 'AUPRC': round(float(average_precision_score(y_true, scores)), 4), 'Accuracy': round(float(accuracy_score(y_true, pred)), 4), 'Precision': round(float(precision_score(y_true, pred, zero_division=0)), 4), 'Recall': round(float(recall_score(y_true, pred, zero_division=0)), 4), 'F1': round(float(f1_score(y_true, pred, zero_division=0)), 4), 'MCC': round(float(matthews_corrcoef(y_true, pred)), 4), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

        def run_cv(X: pd.DataFrame, y: pd.Series, genes: pd.Series):
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            fold_rows, predictions, feature_rows, score_rows, ablation_rows = ([], [], [], [], [])
            member_names = list(MEMBER_DISPLAY)
            for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
                print(f'  Development CV fold {fold}/{N_FOLDS}', flush=True)
                X_outer_train, y_outer_train = (X.iloc[train_index], y.iloc[train_index])
                X_outer_test, y_outer_test = (X.iloc[test_index], y.iloc[test_index])
                fit_index, validation_index = train_test_split(np.arange(len(X_outer_train)), test_size=INNER_VALIDATION_SIZE, stratify=y_outer_train, random_state=SEED + fold)
                X_fit, y_fit = (X_outer_train.iloc[fit_index], y_outer_train.iloc[fit_index])
                X_validation, y_validation = (X_outer_train.iloc[validation_index], y_outer_train.iloc[validation_index])
                threshold_features, _ = select_mi_features(X_fit, y_fit)
                threshold_model = build_mosve()
                threshold_model.fit(X_fit[threshold_features], y_fit)
                validation_scores = threshold_model.predict_proba(X_validation[threshold_features])[:, 1]
                threshold = tune_threshold(y_validation, validation_scores)
                selected_features, ranking = select_mi_features(X_outer_train, y_outer_train)
                model = build_mosve()
                model.fit(X_outer_train[selected_features], y_outer_train)
                member_scores = member_probabilities(model, X_outer_test[selected_features])
                test_scores = average_scores(member_scores, member_names)
                for mode, operating_threshold in [('default_0.50', 0.5), ('inner_tuned_F1', threshold)]:
                    fold_rows.append({'Fold': fold, 'Threshold_Mode': mode, 'Train_Size': int(len(train_index)), 'Test_Size': int(len(test_index)), **metrics(y_outer_test, test_scores, operating_threshold)})
                configurations = {MOSVE_DISPLAY: test_scores}
                for name, scores in member_scores.items():
                    configurations[MEMBER_DISPLAY[name]] = scores
                for label, names in LEAVE_ONE_OUT.items():
                    configurations[label] = average_scores(member_scores, names)
                for config_name, scores in configurations.items():
                    ablation_rows.append({'Fold': fold, 'Configuration': config_name, **metrics(y_outer_test, scores, 0.5)})
                fold_genes = genes.iloc[test_index].values
                fold_labels = y_outer_test.values
                predictions.append(pd.DataFrame({'Fold': fold, 'Gene': fold_genes, 'Label': fold_labels, 'Score': test_scores, 'Tuned_Threshold': threshold, 'Predicted_Default': (test_scores >= 0.5).astype(int), 'Predicted_Tuned': (test_scores >= threshold).astype(int)}))
                for gene, label, ensemble_score in zip(fold_genes, fold_labels, test_scores):
                    score_rows.append({'Fold': fold, 'Gene': gene, 'Label': int(label), 'Model': 'Soft Voting Ensemble', 'Score': float(ensemble_score)})
                for name, scores in member_scores.items():
                    for gene, label, score in zip(fold_genes, fold_labels, scores):
                        score_rows.append({'Fold': fold, 'Gene': gene, 'Label': int(label), 'Model': MEMBER_DISPLAY[name], 'Score': float(score)})
                selected = pd.DataFrame({'Fold': fold, 'Rank': np.arange(1, len(selected_features) + 1), 'Feature': selected_features})
                feature_rows.append(selected)
                ranking.head(MI_K).assign(Fold=fold, Selection='outer_train').to_csv(RESULTS_DIR / f'fold_{fold:02d}_mi_ranking.csv', index=False)
            return (pd.DataFrame(fold_rows), pd.concat(predictions, ignore_index=True), pd.concat(feature_rows, ignore_index=True), pd.DataFrame(score_rows), pd.DataFrame(ablation_rows))

        def evaluate_holdout(X_development, y_development, X_holdout, y_holdout, holdout_genes):
            fit_index, validation_index = train_test_split(np.arange(len(X_development)), test_size=INNER_VALIDATION_SIZE, stratify=y_development, random_state=SEED)
            X_fit, y_fit = (X_development.iloc[fit_index], y_development.iloc[fit_index])
            X_validation, y_validation = (X_development.iloc[validation_index], y_development.iloc[validation_index])
            threshold_features, _ = select_mi_features(X_fit, y_fit)
            threshold_model = build_mosve()
            threshold_model.fit(X_fit[threshold_features], y_fit)
            validation_scores = threshold_model.predict_proba(X_validation[threshold_features])[:, 1]
            threshold = tune_threshold(y_validation, validation_scores)
            final_features, ranking = select_mi_features(X_development, y_development)
            model = build_mosve()
            model.fit(X_development[final_features], y_development)
            scores = model.predict_proba(X_holdout[final_features])[:, 1]
            results = pd.DataFrame([{'Threshold_Mode': 'default_0.50', **metrics(y_holdout, scores, 0.5)}, {'Threshold_Mode': 'inner_tuned_F1', **metrics(y_holdout, scores, threshold)}])
            predictions = pd.DataFrame({'Gene': holdout_genes.values, 'Label': y_holdout.values, 'Score': scores, 'Tuned_Threshold': threshold, 'Predicted_Default': (scores >= 0.5).astype(int), 'Predicted_Tuned': (scores >= threshold).astype(int)})
            return (results, predictions, ranking.head(MI_K))

        def write_figures(cv_predictions: pd.DataFrame, holdout_predictions: pd.DataFrame, holdout_results: pd.DataFrame):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            fpr, tpr, _ = roc_curve(cv_predictions['Label'], cv_predictions['Score'])
            precision, recall, _ = precision_recall_curve(cv_predictions['Label'], cv_predictions['Score'])
            cv_auc = roc_auc_score(cv_predictions['Label'], cv_predictions['Score'])
            cv_auprc = average_precision_score(cv_predictions['Label'], cv_predictions['Score'])
            axes[0].plot(fpr, tpr, color='#007C91', lw=2.4, label=f'MOSVE (AUC = {cv_auc:.4f})')
            axes[0].plot([0, 1], [0, 1], '--', color='#999999', label='Random baseline')
            axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Development-set 10-fold ROC curve')
            axes[0].legend(frameon=False)
            axes[1].plot(recall, precision, color='#C83E3A', lw=2.4, label=f'MOSVE (AUPRC = {cv_auprc:.4f})')
            axes[1].axhline(cv_predictions['Label'].mean(), ls='--', color='#999999', label='Class prevalence')
            axes[1].set(xlabel='Recall', ylabel='Precision', title='Development-set 10-fold precision-recall curve')
            axes[1].legend(frameon=False)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / 'development_cv_roc_pr.png', dpi=220, bbox_inches='tight')
            plt.close(fig)
            tuned = holdout_results.query("Threshold_Mode == 'inner_tuned_F1'").iloc[0]
            matrix = np.array([[tuned.TN, tuned.FP], [tuned.FN, tuned.TP]], dtype=int)
            fig, ax = plt.subplots(figsize=(5.1, 4.5))
            image = ax.imshow(matrix, cmap='Blues')
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_xticks([0, 1], ['Non-driver', 'Driver'])
            ax.set_yticks([0, 1], ['Non-driver', 'Driver'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Observed')
            ax.set_title(f'Reserved-holdout confusion matrix\nROC-AUC={tuned.ROC_AUC:.4f}, AUPRC={tuned.AUPRC:.4f}')
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / 'reserved_holdout_confusion_matrix.png', dpi=220, bbox_inches='tight')
            plt.close(fig)

        def main():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print('=' * 72)
            print('STRICT VERIFIED 4,420-GENE MOSVE EXPERIMENT')
            print('=' * 72)
            cohort, audit, profile = build_strict_cohort()
            validate_bundled_gene_labels(cohort)
            cohort.to_csv(DATA_DIR / 'strict_verified_4420_feature_matrix.csv', index=False)
            audit.to_csv(DATA_DIR / 'strict_verified_4420_label_audit.csv', index=False)
            print(f"Strict cohort: {profile['rows']} genes | positive={profile['positive']} | negative={profile['negative']}")
            y = cohort[TARGET_COL].astype(int).reset_index(drop=True)
            genes = cohort[ID_COL].reset_index(drop=True)
            X = cohort.drop(columns=[ID_COL, TARGET_COL], errors='ignore').select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            excluded_columns = []
            if EXCLUDE_INTOGEN_FEATURES:
                excluded_columns = [column for column in X.columns if 'intogen' in column.lower()]
                X = X.drop(columns=excluded_columns)
            X = X.reset_index(drop=True)
            profile['model_features'] = int(X.shape[1])
            profile['excluded_model_features'] = excluded_columns
            (DATA_DIR / 'strict_verified_4420_profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
            pd.DataFrame({'Feature': X.columns}).to_csv(DATA_DIR / 'model_feature_columns.csv', index=False)
            print(f"Feature matrix: {profile['features']} raw feature columns; {profile['model_features']} model columns")
            if excluded_columns:
                print(f'Excluded source-derived columns: {len(excluded_columns)} IntOGen columns')
            if X.shape[1] < MI_K:
                raise RuntimeError(f'Only {X.shape[1]} numeric features are available; MI K={MI_K} is invalid.')
            X_development, X_holdout, y_development, y_holdout, genes_development, genes_holdout = train_test_split(X, y, genes, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED)
            for value in [X_development, X_holdout, y_development, y_holdout, genes_development, genes_holdout]:
                value.reset_index(drop=True, inplace=True)
            split = pd.concat([pd.DataFrame({'Gene': genes_development, 'Split': 'development', 'Label': y_development}), pd.DataFrame({'Gene': genes_holdout, 'Split': 'reserved_holdout', 'Label': y_holdout})], ignore_index=True)
            split.to_csv(DATA_DIR / 'strict_verified_4420_split_membership.csv', index=False)
            print(f'Development: {len(X_development)} genes ({int(y_development.eq(1).sum())} positive)')
            print(f'Holdout: {len(X_holdout)} genes ({int(y_holdout.eq(1).sum())} positive)')
            print('\nRunning 10-fold development CV ...')
            cv_folds, cv_predictions, selected_features, model_scores, ablation_folds = run_cv(X_development, y_development, genes_development)
            cv_folds.to_csv(RESULTS_DIR / 'development_cv10_fold_metrics.csv', index=False)
            cv_predictions.to_csv(RESULTS_DIR / 'development_cv10_oof_predictions.csv', index=False)
            selected_features.to_csv(RESULTS_DIR / 'development_cv10_selected_features.csv', index=False)
            model_scores.to_csv(RESULTS_DIR / 'development_cv_oof_scores_by_model.csv', index=False)
            ablation_folds.to_csv(RESULTS_DIR / 'development_cv_classifier_ablation_fold_metrics.csv', index=False)
            ablation_summary_rows = []
            full = ablation_folds[ablation_folds['Configuration'] == MOSVE_DISPLAY]
            full_auprc = float(full['AUPRC'].mean())
            for config_name, group in ablation_folds.groupby('Configuration', sort=False):
                ablation_summary_rows.append({'Configuration': config_name, 'ROC_AUC_mean': round(float(group['ROC_AUC'].mean()), 4), 'ROC_AUC_std': round(float(group['ROC_AUC'].std(ddof=1)), 4), 'AUPRC_mean': round(float(group['AUPRC'].mean()), 4), 'AUPRC_std': round(float(group['AUPRC'].std(ddof=1)), 4), 'Accuracy_mean': round(float(group['Accuracy'].mean()), 4), 'Accuracy_std': round(float(group['Accuracy'].std(ddof=1)), 4), 'Precision_mean': round(float(group['Precision'].mean()), 4), 'Precision_std': round(float(group['Precision'].std(ddof=1)), 4), 'F1_mean': round(float(group['F1'].mean()), 4), 'F1_std': round(float(group['F1'].std(ddof=1)), 4), 'Delta_AUPRC_vs_Full': round(float(group['AUPRC'].mean()) - full_auprc, 4)})
            pd.DataFrame(ablation_summary_rows).to_csv(RESULTS_DIR / 'development_cv_classifier_ablation_summary.csv', index=False)
            metric_names = ['ROC_AUC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
            cv_summary_rows = []
            for mode, group in cv_folds.groupby('Threshold_Mode', sort=True):
                row = {'Threshold_Mode': mode, 'N_Folds': len(group)}
                for metric_name in metric_names:
                    row[f'{metric_name}_Mean'] = round(float(group[metric_name].mean()), 4)
                    row[f'{metric_name}_Std'] = round(float(group[metric_name].std(ddof=1)), 4)
                cv_summary_rows.append(row)
            cv_summary = pd.DataFrame(cv_summary_rows)
            cv_summary.to_csv(RESULTS_DIR / 'development_cv10_summary.csv', index=False)
            print('\nRunning reserved 20% holdout evaluation ...')
            holdout_results, holdout_predictions, holdout_features = evaluate_holdout(X_development, y_development, X_holdout, y_holdout, genes_holdout)
            holdout_results.to_csv(RESULTS_DIR / 'reserved_holdout_metrics.csv', index=False)
            holdout_predictions.to_csv(RESULTS_DIR / 'reserved_holdout_predictions.csv', index=False)
            holdout_features.to_csv(RESULTS_DIR / 'reserved_holdout_selected_features.csv', index=False)
            write_figures(cv_predictions, holdout_predictions, holdout_results)
            cv_tuned = cv_summary.query("Threshold_Mode == 'inner_tuned_F1'").iloc[0].to_dict()
            cv_default = cv_summary.query("Threshold_Mode == 'default_0.50'").iloc[0].to_dict()
            holdout_tuned = holdout_results.query("Threshold_Mode == 'inner_tuned_F1'").iloc[0].to_dict()
            holdout_default = holdout_results.query("Threshold_Mode == 'default_0.50'").iloc[0].to_dict()
            summary = {'profile': profile, 'development_size': int(len(X_development)), 'holdout_size': int(len(X_holdout)), 'cv10_default_0.50': cv_default, 'cv10_inner_tuned': cv_tuned, 'holdout_default_0.50': holdout_default, 'holdout_inner_tuned': holdout_tuned, 'model': 'MOSVE: Extra Trees + HistGradientBoosting + Logistic Regression + LightGBM', 'mi_k': MI_K, 'cv_seed': SEED, 'split_seed': SPLIT_SEED}
            (RESULTS_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            report = ['# Strict Verified 4,420-Gene MOSVE Run', '', f"- Cohort: {profile['rows']} genes ({profile['positive']} positive; {profile['negative']} negative)", f'- Development / reserved holdout: {len(X_development)} / {len(X_holdout)} genes', '- Driver sources: CGC, OncoKB, OnGene, NCG', '- Feature selection: mutual information, fit within each training partition (K=200)', '- Ensemble: Extra Trees, HistGradientBoosting, Logistic Regression, LightGBM; soft voting', '', '## Development 10-fold CV (threshold 0.50)', f"- ROC-AUC: {cv_default['ROC_AUC_Mean']:.4f} +/- {cv_default['ROC_AUC_Std']:.4f}", f"- AUPRC: {cv_default['AUPRC_Mean']:.4f} +/- {cv_default['AUPRC_Std']:.4f}", f"- Precision: {cv_default['Precision_Mean']:.4f} +/- {cv_default['Precision_Std']:.4f}", f"- F1: {cv_default['F1_Mean']:.4f} +/- {cv_default['F1_Std']:.4f}", '', '## Development 10-fold CV (inner-tuned threshold)', f"- ROC-AUC: {cv_tuned['ROC_AUC_Mean']:.4f} +/- {cv_tuned['ROC_AUC_Std']:.4f}", f"- AUPRC: {cv_tuned['AUPRC_Mean']:.4f} +/- {cv_tuned['AUPRC_Std']:.4f}", f"- Precision: {cv_tuned['Precision_Mean']:.4f} +/- {cv_tuned['Precision_Std']:.4f}", f"- F1: {cv_tuned['F1_Mean']:.4f} +/- {cv_tuned['F1_Std']:.4f}", '', '## Reserved Holdout (threshold 0.50)', f"- ROC-AUC: {holdout_default['ROC_AUC']:.4f}", f"- AUPRC: {holdout_default['AUPRC']:.4f}", f"- Precision: {holdout_default['Precision']:.4f}", f"- Recall: {holdout_default['Recall']:.4f}", f"- F1: {holdout_default['F1']:.4f}", '', '## Reserved Holdout (inner-tuned threshold)', f"- ROC-AUC: {holdout_tuned['ROC_AUC']:.4f}", f"- AUPRC: {holdout_tuned['AUPRC']:.4f}", f"- Precision: {holdout_tuned['Precision']:.4f}", f"- Recall: {holdout_tuned['Recall']:.4f}", f"- F1: {holdout_tuned['F1']:.4f}"]
            (RESULTS_DIR / 'RUN_REPORT.md').write_text('\n'.join(report), encoding='utf-8')
            print('\n' + '=' * 72)
            print('STRICT FOUR-MODEL MOSVE RUN COMPLETE')
            print('=' * 72)
            print(f"CV10 ROC-AUC: {cv_default['ROC_AUC_Mean']:.4f} +/- {cv_default['ROC_AUC_Std']:.4f}")
            print(f"CV10 AUPRC:    {cv_default['AUPRC_Mean']:.4f} +/- {cv_default['AUPRC_Std']:.4f}")
            print(f"Holdout ROC-AUC: {holdout_default['ROC_AUC']:.4f}")
            print(f"Holdout AUPRC:    {holdout_default['AUPRC']:.4f}")
            print('Classifier ablation (threshold 0.50):')
            for row in ablation_summary_rows:
                print(f"  {row['Configuration']}: ROC-AUC {row['ROC_AUC_mean']:.4f}+/-{row['ROC_AUC_std']:.4f}  AUPRC {row['AUPRC_mean']:.4f}+/-{row['AUPRC_std']:.4f}  dAUPRC {row['Delta_AUPRC_vs_Full']:+.4f}")
            print(f'Results: {RESULTS_DIR}')
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

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
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'candidates'])
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
    {'train': _run_train, 'candidates': _run_candidates}[args.task]([])

if __name__ == '__main__':
    main()
