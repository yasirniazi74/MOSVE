from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def validate_release_matches(cohort, release_path):
    import pandas as pd
    from pathlib import Path
    release_path = Path(release_path)
    if release_path.is_dir():
        release = pd.concat([
            pd.read_csv(release_path / "positive_genes.csv").assign(Label=1),
            pd.read_csv(release_path / "negative_genes.csv").assign(Label=0),
        ], ignore_index=True)
    else:
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
        LABEL_LEDGER = WORK_DIR / '02_data' / 'data_files' / 'all_gene_labels_combined.csv'
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
            release_dir = WORK_DIR / '02_data' / 'data_files'
            release = pd.concat([
                pd.read_csv(release_dir / 'positive_genes.csv').assign(Label=1),
                pd.read_csv(release_dir / 'negative_genes.csv').assign(Label=0),
            ], ignore_index=True)
            release[ID_COL] = release[ID_COL].astype(str).str.strip().str.upper()
            release[TARGET_COL] = pd.to_numeric(release[TARGET_COL], errors='raise').astype(int)
            if release[ID_COL].duplicated().any():
                raise ValueError(f'Bundled gene-label release has duplicate symbols: {release_dir}')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train'])
    arguments = _sys.argv[1:]
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser.print_help()
        return
    args = parser.parse_args(arguments[:1])
    if len(arguments) > 1:
        parser.error(f'{args.task} does not accept extra arguments')
    _run_train([])

if __name__ == '__main__':
    main()
