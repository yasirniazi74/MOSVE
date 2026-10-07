from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def _run_unseen_gene(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/cptac_unseen_gene.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        import json
        import numpy as np, pandas as pd
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        ROOT = Path(__file__).resolve().parents[2]
        OUT = ROOT / '05_external_validation' / '18_cptac_unseen_gene_validation'
        OUT.mkdir(parents=True, exist_ok=True)
        TRAIN = ROOT / '04_strict_verified_4420_experiment' / 'data' / 'strict_verified_4420_feature_matrix.csv'
        LAB = ROOT / '05_external_validation' / '05_labels' / 'pcawg_consensus_driver_elements.csv'
        CPTAC = ROOT / '05_external_validation' / '16_full_locked_cptac_feature_audit'
        SEED = 42
        K = 200
        THRESHOLD = 0.5

        def pipe(est, scale=False):
            s = [('imputer', SimpleImputer(strategy='median'))]
            if scale:
                s.append(('scaler', StandardScaler()))
            s.append(('model', est))
            return Pipeline(s)

        def model():
            return VotingClassifier(estimators=[('ExtraTrees', pipe(ExtraTreesClassifier(n_estimators=700, min_samples_split=5, min_samples_leaf=1, max_features=0.5, class_weight='balanced', random_state=SEED, n_jobs=1))), ('HistGradientBoosting', pipe(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.1, random_state=SEED))), ('LogisticRegression', pipe(LogisticRegression(C=0.06951927961775606, solver='liblinear', penalty='l2', max_iter=7000, class_weight='balanced', random_state=SEED), True)), ('LightGBM', pipe(LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6, reg_lambda=0.0, random_state=SEED, n_jobs=1, verbose=-1)))], voting='soft', n_jobs=1)

        def choose_controls(train, pos, target_n):
            pool = train[train.Label.eq(0) & ~train.Gene.isin(pos)].copy()
            cols = ['Seq_Length', 'GC_Content', 'Entropy_2mer', 'SNV_Count']
            a = train.set_index('Gene').loc[sorted(pos), cols].copy()
            b = pool.set_index('Gene')[cols].copy()
            both = pd.concat([a, b])
            mu = both.mean()
            sd = both.std().replace(0, 1)
            used = set()
            selected = []
            ordered = sorted(pos)
            while len(selected) < target_n:
                added = 0
                for gene in ordered:
                    if len(selected) >= target_n:
                        break
                    delta = ((b - mu) / sd - (a.loc[gene] - mu) / sd).pow(2).sum(axis=1)
                    cand = delta[~delta.index.isin(used)].sort_values().index[0]
                    selected.append(cand)
                    used.add(cand)
                    added += 1
                if not added:
                    raise RuntimeError('Insufficient unique reference controls')
            return selected

        def metric(y, s):
            p = (s >= THRESHOLD).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
            return {'N': int(len(y)), 'Positive': int(y.sum()), 'Negative': int((1 - y).sum()), 'Threshold': THRESHOLD, 'ROC_AUC': float(roc_auc_score(y, s)), 'AUPRC': float(average_precision_score(y, s)), 'Accuracy': float(accuracy_score(y, p)), 'Precision': float(precision_score(y, p, zero_division=0)), 'Recall': float(recall_score(y, p, zero_division=0)), 'F1': float(f1_score(y, p, zero_division=0)), 'MCC': float(matthews_corrcoef(y, p)), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

        def ci(y, s):
            rng = np.random.default_rng(SEED)
            ix1 = np.flatnonzero(y == 1)
            ix0 = np.flatnonzero(y == 0)
            a = []
            p = []
            for _ in range(1000):
                ix = np.r_[rng.choice(ix1, len(ix1), True), rng.choice(ix0, len(ix0), True)]
                a.append(roc_auc_score(y[ix], s[ix]))
                p.append(average_precision_score(y[ix], s[ix]))
            return {'ROC_AUC_CI95': [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))], 'AUPRC_CI95': [float(np.quantile(p, 0.025)), float(np.quantile(p, 0.975))]}

        def main():
            tr = pd.read_csv(TRAIN)
            tr.Gene = tr.Gene.astype(str).str.upper()
            feats = [x for x in tr if x not in ['Gene', 'Label']]
            l = pd.read_csv(LAB)
            results = []
            manifest = {}
            for cohort, tissue in [('CCRCC', 'Kidney-RCC'), ('HNSCC', 'Head-SCC')]:
                pos = set(l.loc[l.Tissue.fillna('').str.contains(tissue, regex=False), 'Gene'].astype(str).str.upper()) & set(tr.Gene)
                original_ratio = tr.Label.eq(0).sum() / tr.Label.eq(1).sum()
                target_controls = round(len(pos) * original_ratio)
                neg = choose_controls(tr, pos, target_controls)
                test_genes = sorted(pos) + neg
                train = tr[~tr.Gene.isin(test_genes)].copy()
                X = train[feats].replace([np.inf, -np.inf], np.nan)
                imp = SimpleImputer(strategy='median').fit_transform(X)
                rank = pd.Series(mutual_info_classif(MinMaxScaler().fit_transform(imp), train.Label, random_state=SEED), index=feats).sort_values(ascending=False)
                sel = rank.head(K).index.tolist()
                m = model()
                m.fit(train[sel], train.Label)
                ext = pd.read_csv(CPTAC / f'{cohort.lower()}_locked_gene_by_305_model_features.csv')
                ext.Gene = ext.Gene.astype(str).str.upper()
                test = ext.set_index('Gene').loc[test_genes].reset_index()
                y = np.array([int(g in pos) for g in test.Gene])
                score = m.predict_proba(test[sel])[:, 1]
                out = test[['Gene']].copy()
                out['External_Label'] = y
                out['Reference_Class'] = ['PCAWG_' + tissue + '_positive' if q else 'matched_strict_reference_background' for q in y]
                out['MOSVE_Score'] = score
                out['Predicted_Driver_0_50'] = (score >= THRESHOLD).astype(int)
                out['Selected_Feature_NA_Count'] = test[sel].isna().sum(axis=1)
                component_rows = []
                for name, estimator in m.named_estimators_.items():
                    member_score = estimator.predict_proba(test[sel])[:, 1]
                    out[name + '_Score'] = member_score
                    component_rows.append({'Cohort': cohort, 'Algorithm': name, **metric(y, member_score), **ci(y, member_score), 'External_Positives': len(pos), 'Matched_Controls': len(neg), 'TCGA_Training_Genes': len(train), 'Selected_Features': len(sel)})
                pd.DataFrame(component_rows).to_csv(OUT / f'{cohort.lower()}_base_learner_metrics.csv', index=False)
                out.sort_values('MOSVE_Score', ascending=False).to_csv(OUT / f'{cohort.lower()}_unseen_gene_cptac_predictions.csv', index=False)
                met = {**metric(y, score), **ci(y, score), 'Cohort': cohort, 'External_Positives': len(pos), 'Matched_Controls': len(neg), 'TCGA_Training_Genes': len(train), 'TCGA_Training_Positives': int(train.Label.sum()), 'Selected_Features': len(sel), 'Test_Genes_Excluded_From_Training': len(test_genes), 'Test_Label_Overlap_With_Original_Training_Positive': int(tr[tr.Gene.isin(pos)].Label.sum())}
                results.append(met)
                pd.DataFrame({'Feature': sel, 'MI_train_only': rank.loc[sel].values}).to_csv(OUT / f'{cohort.lower()}_training_only_selected_features.csv', index=False)
                manifest[cohort] = met
            pd.DataFrame(results).to_csv(OUT / 'unseen_gene_external_metrics.csv', index=False)
            pd.concat([pd.read_csv(OUT / f'{c}_base_learner_metrics.csv') for c in ['ccrcc', 'hnscc']], ignore_index=True).to_csv(OUT / 'base_learner_external_metrics.csv', index=False)
            json.dump({'design': 'TCGA features/train only; CPTAC features/test only; all test genes excluded before MI selection, imputation and model fitting', 'labels': 'PCAWG tissue-specific positives; deterministic matched strict-reference-background controls at the original strict MOSVE negative:positive ratio (2731:1689); non-called genes are proxy controls, not confirmed biological negatives', 'settings': 'Fixed four-model strict MOSVE architecture, 200 MI features selected on training only, threshold 0.50', 'results': manifest}, open(OUT / 'manifest.json', 'w'), indent=2)
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv

def _run_cv10(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/cptac_cv10.py')
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
        INP = ROOT / '05_external_validation/16_full_locked_cptac_feature_audit'
        LABEL = ROOT / '04_strict_verified_4420_experiment/data/strict_verified_4420_feature_matrix.csv'
        REMOVE = ROOT / '05_external_validation/19_cptac_only_gene_split_benchmark/removed_62_all_na_features.csv'
        OUT = ROOT / '05_external_validation/28_cptac_same_cohort_cv10'
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
            labels = pd.read_csv(LABEL, usecols=['Gene', 'Label'])
            labels.Gene = labels.Gene.astype(str).str.upper()
            remove = pd.read_csv(REMOVE).iloc[:, 0].tolist()
            allrows = []
            folds = []
            for cohort in ['CCRCC', 'HNSCC']:
                d = pd.read_csv(INP / f'{cohort.lower()}_locked_gene_by_305_model_features.csv')
                d.Gene = d.Gene.astype(str).str.upper()
                d = d.merge(labels, on='Gene', how='inner', validate='one_to_one')
                features = [c for c in d if c not in ['Gene', 'Label'] and c not in remove]
                y = d.Label.to_numpy()
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
                names = ['ExtraTrees', 'HistGradientBoosting', 'LogisticRegression', 'LightGBM', 'MOSVE ensemble']
                oof = {n: np.full(len(d), np.nan) for n in names}
                for fold, (tr, te) in enumerate(skf.split(d, y), 1):
                    m = build().fit(d.iloc[tr][features], y[tr])
                    for n, e in list(m.named_estimators_.items()) + [('MOSVE ensemble', m)]:
                        s = e.predict_proba(d.iloc[te][features])[:, 1]
                        oof[n][te] = s
                        folds.append({'Cohort': cohort, 'Fold': fold, 'Model': n, 'N': len(te), **met(y[te], s)})
                p = d[['Gene', 'Label']].copy()
                for n, s in oof.items():
                    allrows.append({'Cohort': cohort, 'Model': n, **met(y, s)})
                    p[n.replace(' ', '_') + '_OOF_Score'] = s
                p.to_csv(OUT / f'{cohort.lower()}_oof_predictions.csv', index=False)
            pd.DataFrame(allrows).to_csv(OUT / 'oof_metrics.csv', index=False)
            pd.DataFrame(folds).to_csv(OUT / 'fold_metrics.csv', index=False)
            (OUT / 'manifest.json').write_text(json.dumps({'features': 243, 'removed_all_na_features': 62, 'cv': 'stratified 10-fold by gene, seed 42', 'threshold': T, 'labels': 'existing curated gene-label ledger; this is not independent-label validation'}, indent=2))
            print(pd.DataFrame(allrows).round(4).to_string(index=False))
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv


def main():
    parser = argparse.ArgumentParser(description='Run CPTAC external validation analyses.')
    parser.add_argument('task', choices=['unseen_gene', 'cv10'])
    arguments = _sys.argv[1:]
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser.print_help()
        return
    args = parser.parse_args(arguments[:1])
    if len(arguments) > 1:
        parser.error(f'{args.task} does not accept extra arguments')
    {'unseen_gene': _run_unseen_gene, 'cv10': _run_cv10}[args.task]([])

if __name__ == '__main__':
    main()
