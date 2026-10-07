from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def _run_download(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/cptac_download.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        import argparse
        import concurrent.futures
        import csv
        import hashlib
        import json
        import re
        import time
        import urllib.error
        import urllib.parse
        import urllib.request
        from datetime import datetime, timezone
        from html.parser import HTMLParser
        from pathlib import Path
        WORK = Path(__file__).resolve().parents[2]
        DEFAULT_OUTPUT = WORK / '05_external_validation/09_cptac_downloads'
        FIGSHARE_API = 'https://api.figshare.com/v2/articles/27993248/versions/1'
        CORE = {'RNAseq (Gene level, Tumor)', 'Methylation (Tumor)', 'CNV (log2 ratio)', 'CNV (GISTIC level)', 'Mutation MAF', 'Meta table'}

        def request(url, headers=None):
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'MOSVE-research-downloader/1.0', **(headers or {})}), timeout=60)

        class DownloadTable(HTMLParser):

            def __init__(self):
                super().__init__()
                self.rows, self.cells, self.links, self.cell = ([], [], [], None)

            def handle_starttag(self, tag, attrs):
                if tag == 'tr':
                    self.cells, self.links = ([], [])
                elif tag == 'td':
                    self.cell = []
                elif tag == 'a':
                    href = dict(attrs).get('href')
                    if href:
                        self.links.append(href)

            def handle_data(self, data):
                if self.cell is not None:
                    self.cell.append(data)

            def handle_endtag(self, tag):
                if tag == 'td' and self.cell is not None:
                    self.cells.append(' '.join(''.join(self.cell).split()))
                    self.cell = None
                elif tag == 'tr' and self.cells and self.links:
                    self.rows.append((self.cells[0], self.links[0]))

        def checksums(path):
            sha, md5 = (hashlib.sha256(), hashlib.md5())
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    sha.update(chunk)
                    md5.update(chunk)
            return (sha.hexdigest(), md5.hexdigest())

        def validate_table(path):
            with path.open('rb') as handle:
                head = handle.read(65536)
            if not head or b'<html' in head.lower() or b'<!doctype html' in head.lower():
                raise ValueError(f'Empty file or HTML error page: {path}')
            text = head.decode('utf-8-sig')
            lines = [line for line in text.splitlines() if line and (not line.startswith('#'))]
            if not lines or ('\t' not in lines[0] and ',' not in lines[0]):
                raise ValueError(f'No tabular header: {path}')
            if path.suffix == '.maf' and 'Hugo_Symbol' not in text:
                raise ValueError(f'Missing MAF gene header: {path}')
            return lines[0].count('\t' if '\t' in lines[0] else ',') + 1

        def discover(args):
            jobs = []
            labels = CORE | ({'RNAseq (Gene level, Normal)', 'Methylation (Normal)'} if args.include_normals else set())
            for cohort in args.cohorts:
                page = f'https://www.linkedomics.org/data_download/CPTAC-pancan-{cohort}/'
                with request(page) as response:
                    parser = DownloadTable()
                    parser.feed(response.read().decode('utf-8'))
                found = set()
                for label, href in parser.rows:
                    if label not in labels:
                        continue
                    url = urllib.parse.urljoin(page, href)
                    filename = Path(urllib.parse.urlparse(url).path).name
                    jobs.append(dict(dataset=f'CPTAC-{cohort}', modality=label, url=url, source_page=page, path=str(args.output / cohort / filename)))
                    found.add(label)
                if labels - found:
                    raise ValueError(f'Missing download links for {cohort}: {sorted(labels - found)}')
            if 'HNSCC' in args.cohorts and (not args.skip_probe_methylation):
                page = 'https://www.linkedomics.org/data_download/CPTAC-HNSCC/'
                with request(page) as response:
                    parser = DownloadTable()
                    parser.feed(response.read().decode('utf-8'))
                matches = [(label, href) for label, href in parser.rows if label == 'Methylation (Probe level, Tumor)']
                if len(matches) != 1:
                    raise ValueError('Cannot uniquely locate HNSCC probe methylation')
                label, href = matches[0]
                url = urllib.parse.urljoin(page, href)
                jobs.append(dict(dataset='CPTAC-HNSCC-original', modality=label, url=url, source_page=page, path=str(args.output / 'HNSCC_probe_level' / Path(urllib.parse.urlparse(url).path).name)))
            if not args.without_depmap:
                with request(FIGSHARE_API) as response:
                    article = json.load(response)
                wanted = {'CRISPRGeneEffect.csv', 'Model.csv'}
                selected = [f for f in article['files'] if f['name'] in wanted]
                if len(selected) != len(wanted):
                    raise ValueError('Pinned DepMap release is missing required files')
                for file in selected:
                    jobs.append(dict(dataset='DepMap-24Q4-v1', modality=file['name'], url=file['download_url'], source_page=article['url_public_html'], doi=article['doi'], expected_bytes=file['size'], expected_md5=file['computed_md5'], path=str(args.output / 'DepMap_24Q4' / file['name'])))
            return jobs

        def download_ranges(job, part):
            total = job['expected_bytes']
            prefix = part.stat().st_size if part.exists() else 0
            if prefix > total:
                raise ValueError('Partial file exceeds expected release size')
            block = 16 * 1024 * 1024
            ranges = [(start, min(start + block, total) - 1) for start in range(prefix, total, block)]

            def segment(bounds):
                start, end = bounds
                target = part.with_name(part.name + f'.range-{start}-{end}')
                needed = end - start + 1
                for attempt in range(4):
                    offset = target.stat().st_size if target.exists() else 0
                    if offset == needed:
                        return target
                    if offset > needed:
                        raise ValueError(f'Oversize segment: {target}')
                    try:
                        with request(job['url'], {'Range': f'bytes={start + offset}-{end}'}) as response:
                            expected = f'bytes {start + offset}-{end}/{total}'
                            if response.status != 206 or response.headers.get('Content-Range') != expected:
                                raise ValueError('Server did not honor the requested byte range')
                            with target.open('ab') as handle:
                                while True:
                                    chunk = response.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    handle.write(chunk)
                        if target.stat().st_size != needed:
                            raise ValueError('Incomplete range response')
                        print(f"RANGE DONE {Path(job['path']).name}: {start}-{end}", flush=True)
                        return target
                    except (OSError, ValueError):
                        if attempt == 3:
                            raise
                        time.sleep(2 * (attempt + 1))
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                segments = list(pool.map(segment, ranges))
            with part.open('ab') as output:
                for target in segments:
                    with target.open('rb') as handle:
                        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                            output.write(chunk)
            if part.stat().st_size != total:
                raise ValueError('Assembled range download has the wrong size')

        def download(job, previous):
            job = dict(job)
            path = Path(job['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + '.part')
            try:
                if path.exists():
                    sha, md5 = checksums(path)
                    old = previous.get(str(path), {})
                    valid = old.get('sha256') == sha and old.get('url') == job['url']
                    valid |= bool(job.get('expected_md5') and job['expected_md5'] == md5)
                    if not valid:
                        raise ValueError(f'Existing file is unverified or differs: {path}')
                    for key in ('downloaded_utc', 'etag', 'last_modified'):
                        if key in old:
                            job[key] = old[key]
                    job.update(status='verified_existing', bytes=path.stat().st_size, sha256=sha, md5=md5, header_columns=validate_table(path))
                    print(f"VERIFIED {job['dataset']}: {path.name}", flush=True)
                    return job
                print(f"START {job['dataset']}: {path.name}", flush=True)
                if job.get('expected_bytes', 0) > 100 * 1024 * 1024:
                    download_ranges(job, part)
                    sha, md5 = checksums(part)
                    if md5 != job['expected_md5']:
                        raise ValueError('Assembled MD5 does not match publisher checksum')
                    columns = validate_table(part)
                    part.rename(path)
                    for segment_path in path.parent.glob(path.name + '.part.range-*'):
                        if re.fullmatch(re.escape(path.name) + '\\.part\\.range-\\d+-\\d+', segment_path.name):
                            segment_path.unlink()
                    job.update(status='downloaded', bytes=path.stat().st_size, sha256=sha, md5=md5, header_columns=columns, downloaded_utc=datetime.now(timezone.utc).isoformat())
                    print(f"DONE {path.name}: {job['bytes'] / 1000000.0:.1f} MB (MD5 verified)", flush=True)
                    return job
                for attempt in range(1, 5):
                    try:
                        offset = part.stat().st_size if part.exists() else 0
                        headers = {'Range': f'bytes={offset}-'} if offset else {}
                        with request(job['url'], headers) as response:
                            resumed = response.status == 206 and offset > 0
                            if resumed:
                                content_range = response.headers.get('Content-Range', '')
                                match = re.fullmatch('bytes (\\d+)-(\\d+)/(\\d+)', content_range)
                                if not match or int(match[1]) != offset:
                                    raise ValueError(f'Invalid Content-Range: {content_range}')
                                total = int(match[3])
                            else:
                                offset = 0
                                length = response.headers.get('Content-Length')
                                total = int(length) if length else None
                            job['etag'] = response.headers.get('ETag')
                            job['last_modified'] = response.headers.get('Last-Modified')
                            with part.open('ab' if resumed else 'wb') as handle:
                                last_report = time.monotonic()
                                while True:
                                    chunk = response.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    handle.write(chunk)
                                    offset += len(chunk)
                                    if time.monotonic() - last_report > 20:
                                        print(f'PROGRESS {path.name}: {offset / 1000000.0:.1f} MB', flush=True)
                                        last_report = time.monotonic()
                        if total is not None and part.stat().st_size != total:
                            raise ValueError('Incomplete response body')
                        if job.get('expected_bytes') and part.stat().st_size != job['expected_bytes']:
                            raise ValueError('Size does not match release metadata')
                        sha, md5 = checksums(part)
                        if job.get('expected_md5') and md5 != job['expected_md5']:
                            raise ValueError('MD5 does not match publisher checksum')
                        columns = validate_table(part)
                        if path.suffix == '.maf':
                            with part.open(encoding='utf-8-sig') as handle:
                                if 'Hugo_Symbol' not in handle.read(65536):
                                    raise ValueError('Missing MAF gene header')
                        part.rename(path)
                        job.update(status='downloaded', bytes=path.stat().st_size, sha256=sha, md5=md5, header_columns=columns, downloaded_utc=datetime.now(timezone.utc).isoformat())
                        print(f"DONE {path.name}: {job['bytes'] / 1000000.0:.1f} MB", flush=True)
                        return job
                    except (OSError, ValueError) as error:
                        if attempt == 4:
                            raise
                        if isinstance(error, urllib.error.HTTPError) and error.code == 416:
                            part.rename(part.with_name(part.name + f'.saved-{time.time_ns()}'))
                        print(f'RETRY {attempt}/4 {path.name}: {error}', flush=True)
                        time.sleep(attempt * 2)
            except Exception as error:
                job.update(status='failed', error=str(error))
                print(f'FAILED {path.name}: {error}', flush=True)
                return job

        def local_references():
            roots = WORK / '02_data' / 'raw_sources' / 'DriverGenesData'
            targets = {'STRING v12.0 full links': roots / '04_Biological_Knowledge/9606.protein.links.full.v12.0.txt/9606.protein.links.full.v12.0.txt', 'STRING v12.0 aliases': roots / '02_Gene_Mapping/9606.protein.aliases.v12.0.txt/9606.protein.aliases.v12.0.txt', 'Training methylation 450K annotation': WORK / '02_data' / 'raw_sources' / 'humanmethylation450_15017482_v1-2.csv'}
            records = []
            for name, path in targets.items():
                row = {'resource': name, 'path': str(path), 'exists': path.is_file()}
                if path.is_file():
                    row.update(bytes=path.stat().st_size, sha256=checksums(path)[0])
                records.append(row)
            return records

        def main():
            parser = argparse.ArgumentParser(description='Download the configured CPTAC source files.')
            parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
            parser.add_argument('--cohorts', nargs='+', choices=['HNSCC', 'CCRCC'], default=['HNSCC', 'CCRCC'])
            parser.add_argument('--include-normals', action='store_true')
            parser.add_argument('--without-depmap', action='store_true')
            parser.add_argument('--skip-probe-methylation', action='store_true')
            parser.add_argument('--workers', type=int, choices=range(1, 5), default=3)
            args = parser.parse_args()
            args.output = args.output.resolve()
            args.output.mkdir(parents=True, exist_ok=True)
            manifest_path = args.output / 'download_manifest.json'
            previous = {}
            if manifest_path.exists():
                previous = {r['path']: r for r in json.loads(manifest_path.read_text())['files']}
            jobs = discover(args)
            report = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=[], local_references=local_references(), notes=['Source downloads only; no MOSVE matrix or validation results generated.', 'CPTAC pan-cancer freeze v1.2; audit TCGA overlap and modality sample IDs before use.', 'Original HNSCC probe data is a separate release: match samples and array annotations before combining.', 'CCRCC gene-level methylation cannot establish promoter-feature compatibility.', 'DepMap 24Q4 v1 is a pinned archival release, not the current portal release.', 'STRING v12.0 is reused locally; network features are not independent cohort evidence.', 'PCAWG already exists under 05_external_validation/02_raw_data; METABRIC is optional and not downloaded.'])

            def save():
                temp = manifest_path.with_suffix('.json.tmp')
                temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
                temp.replace(manifest_path)
            selected_paths = {j['path'] for j in jobs}
            report['files'] = [r for path, r in previous.items() if path not in selected_paths]
            report['files'] += [previous.get(j['path'], {**j, 'status': 'pending'}) for j in jobs]
            save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download, j, previous) for j in jobs]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    report['files'] = [record if row['path'] == record['path'] else row for row in report['files']]
                    save()
            selected_records = [r for r in report['files'] if r['path'] in selected_paths]
            failures = sum((r['status'] == 'failed' for r in selected_records))
            report['summary'] = dict(total_files=len(jobs), failed=failures, verified_bytes=sum((r.get('bytes', 0) for r in selected_records if r['status'] != 'failed')))
            save()
            with (args.output / 'download_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['dataset', 'modality', 'status', 'bytes', 'sha256', 'url', 'path', 'error'], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report['files'])
            print(json.dumps(report['summary']), flush=True)
            print(f'Manifest: {manifest_path}', flush=True)
            return 1 if failures else 0
        if __name__ == '__main__':
            raise SystemExit(main())
    finally:
        _sys.argv = _runner_previous_argv

def _run_build(forwarded_args):
    __file__ = str(ROOT / '05_external_validation/_runtime/cptac_features.py')
    __name__ = '__main__'
    _runner_previous_argv = _sys.argv
    _sys.argv = [__file__] + forwarded_args
    try:
        from pathlib import Path
        from collections import Counter, defaultdict
        from math import log2
        import hashlib, json, re
        import numpy as np
        import pandas as pd
        ROOT = Path(__file__).resolve().parents[2]
        RAW = ROOT / '05_external_validation' / '09_cptac_downloads'
        OUT = ROOT / '05_external_validation' / '16_full_locked_cptac_feature_audit'
        OUT.mkdir(parents=True, exist_ok=True)
        MODEL = ROOT / '04_strict_verified_4420_experiment' / 'data' / 'model_feature_columns.csv'
        TRAIN = ROOT / '04_strict_verified_4420_experiment' / 'data' / 'strict_verified_4420_feature_matrix.csv'
        QC = ROOT / '05_external_validation' / '12_methylation_mapping_qc'
        METH = ROOT / '05_external_validation' / '15_locked_probe_methylation_features'
        DATA = ROOT / '02_data' / 'raw_sources' / 'DriverGenesData'
        FASTA = DATA / '03_Multiomics' / 'DNA_Sequence' / 'Cosmic_Genes_Fasta_v104_GRCh37' / 'Cosmic_Genes_v104_GRCh37.fasta' / 'Cosmic_Genes_v104_GRCh37.fasta'
        CLIN = DATA / '03_Multiomics' / 'Pathogenicity' / 'variant_summary.txt'
        PATHWAYS = DATA / '04_Biological_Knowledge' / 'Human_GOBP_AllPathways_noPFOCR_no_GO_iea_February_03_2026_symbol.txt'
        HGNC = DATA / '02_Gene_Mapping' / 'hgnc_complete_set.txt'

        def sha(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(1 << 20), b''):
                    h.update(b)
            return h.hexdigest()

        def genes():
            return pd.read_csv(TRAIN, usecols=['Gene'])['Gene'].str.upper().drop_duplicates().tolist()

        def ensembl_map():
            x = pd.read_csv(HGNC, sep='\t', low_memory=False)
            d = {}
            for _, r in x.iterrows():
                s = str(r.get('symbol', '')).upper()
                e = str(r.get('ensembl_gene_id', '')).split('.')[0]
                if s and s != 'NAN' and e and (e != 'NAN'):
                    d[e] = s
            return d

        def clean_index(x, mp):
            x = x.copy()
            old = x.index.astype(str).str.split('.').str[0]
            x.index = pd.Index([mp.get(v, v).upper() for v in old])
            return x.groupby(level=0).mean()

        def seq_features(target):
            seqs = {}
            g = None
            a = []
            for ln in open(FASTA, encoding='utf-8'):
                if ln.startswith('>'):
                    if g:
                        seqs.setdefault(g, ''.join(a))
                    g = ln[1:].split()[0].upper()
                    a = []
                else:
                    a.append(ln.strip().upper())
            if g:
                seqs.setdefault(g, ''.join(a))
            rows = []
            hydro = set('AILMFWV')
            polar = set('STNQY')
            charged = set('KRDE')
            aromatic = set('FWY')
            codons = [''.join(x) for x in []]
            table = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T', 'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R', 'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P', 'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A', 'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G', 'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L', 'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*', 'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W'}
            for gene in target:
                s = seqs.get(gene)
                if not s:
                    continue
                L = max(len(s), 1)
                A = s.count('A')
                C = s.count('C')
                G = s.count('G')
                T = s.count('T')
                aa = ''.join((table.get(s[i:i + 3], 'X') for i in range(0, L - 2, 3)))
                dc = Counter((s[i:i + 2] for i in range(L - 1)))
                tri = Counter((s[i:i + 3] for i in range(L - 2)))
                r = {'Gene': gene, 'Seq_Length': L, 'GC_Content': (G + C) / L, 'AT_Content': (A + T) / L, 'GC_Skew': (G - C) / max(1, G + C), 'AT_Skew': (A - T) / max(1, A + T), 'Entropy_1mer': -sum((v / L * log2(v / L) for v in Counter(s).values())), 'Entropy_2mer': -sum((v / max(L - 1, 1) * log2(v / max(L - 1, 1)) for v in dc.values())), 'CpG_Density': s.count('CG') / L, 'Protein_Length': len(aa), 'Hydrophobic_Ratio': sum((z in hydro for z in aa)) / max(1, len(aa)), 'Polar_Ratio': sum((z in polar for z in aa)) / max(1, len(aa)), 'Charged_Ratio': sum((z in charged for z in aa)) / max(1, len(aa)), 'Aromatic_Ratio': sum((z in aromatic for z in aa)) / max(1, len(aa)), 'Stop_Codon_Fraction': aa.count('*') / max(1, len(aa)), 'Seq_Length_new': L, 'GC_Content_new': (G + C) / L, 'AT_Content_new': (A + T) / L, 'GC1': sum((s[i] in 'GC' for i in range(0, L, 3))) / max(1, len(s[0::3])), 'GC2': sum((s[i] in 'GC' for i in range(1, L, 3))) / max(1, len(s[1::3])), 'GC3': sum((s[i] in 'GC' for i in range(2, L, 3))) / max(1, len(s[2::3])), 'CpG_OE': s.count('CG') / max(C * G / L, 1e-300)}
                for k, v in dc.items():
                    r['di_' + k] = v / max(L - 1, 1)
                for k, v in tri.items():
                    r['k3_' + k] = v / max(L - 2, 1)
                rows.append(r)
            return pd.DataFrame(rows)

        def expression_cnv(cohort, locked, mp):
            d = RAW / cohort
            expr = clean_index(pd.read_csv(next(d.glob('*RNAseq*Tumor.txt')), sep='\t', index_col=0), mp)
            expr = expr[[x for x in locked if x in expr.columns]].apply(pd.to_numeric, errors='coerce')
            cnv = clean_index(pd.read_csv(next(d.glob('*CNV_gene_ratio_log2.txt')), sep='\t', index_col=0), mp)
            cnv = cnv[[x for x in locked if x in cnv.columns]].apply(pd.to_numeric, errors='coerce')
            e = pd.DataFrame({'Gene': expr.index, 'Expr_Mean': expr.mean(1), 'Expr_STD': expr.std(1), 'Expr_Median': expr.median(1), 'Expr_CV': expr.std(1) / (expr.mean(1).abs() + 1e-06)})
            c = pd.DataFrame({'Gene': cnv.index, 'CNV_Mean': cnv.mean(1), 'CNV_STD': cnv.std(1), 'CNV_Gain_Ratio': (cnv > 0).mean(1), 'CNV_Loss_Ratio': (cnv < 0).mean(1), 'CNV_Amplification_Ratio': (cnv >= 1).mean(1), 'CNV_Deletion_Ratio': (cnv <= -1).mean(1)})
            return (e.reset_index(drop=True), c.reset_index(drop=True), {'expression_samples': len(expr.columns), 'cnv_samples': len(cnv.columns)})

        def mutation(cohort, locked):
            m = pd.read_csv(RAW / cohort / f'{cohort}_somatic_mutation.maf', sep='\t', low_memory=False)
            m['Gene'] = m['Hugo_Symbol'].astype(str).str.upper()
            m = m[m.Tumor_Sample_Barcode.astype(str).isin(locked)].copy()
            vc = m.Variant_Classification.astype(str)
            n = m.groupby('Gene').size()
            out = pd.DataFrame({'Gene': n.index, 'SNV_Count': n.values})
            for nm, mask in [('Missense', vc.eq('Missense_Mutation')), ('Nonsense', vc.eq('Nonsense_Mutation')), ('Frameshift', vc.str.contains('Frame', na=False)), ('Silent', vc.eq('Silent')), ('Splice_Site', vc.eq('Splice_Site'))]:
                out[nm] = m[mask].groupby('Gene').size().reindex(out.Gene, fill_value=0).values
            out['Patients_Affected'] = m.groupby('Gene').Tumor_Sample_Barcode.nunique().reindex(out.Gene).values
            ns = out.Missense + out.Nonsense + out.Frameshift + out.Splice_Site
            tr = out.Nonsense + out.Frameshift
            den = out.SNV_Count.clip(lower=1)
            out['NonSilent_Ratio'] = ns / den
            out['Truncating_Ratio'] = tr / den
            out['Pathogenicity_Score'] = tr + ns
            out['High_Impact_Ratio'] = tr / den
            out['Missense_Ratio'] = out.Missense / den
            out['Silent_Ratio'] = out.Silent / den
            out['Mutation_Impact_Score'] = 2 * tr + ns
            out['Mean_Tumor_VAF'] = m.groupby('Gene').TumorVAF.mean().reindex(out.Gene).values
            out['Max_Tumor_VAF'] = m.groupby('Gene').TumorVAF.max().reindex(out.Gene).values
            out['SNP_Fraction'] = m.assign(z=m.Variant_Type.eq('SNP')).groupby('Gene').z.mean().reindex(out.Gene).values
            out['Indel_Fraction'] = m.assign(z=m.Variant_Type.str.contains('INS|DEL', regex=True, na=False)).groupby('Gene').z.mean().reindex(out.Gene).values
            out['SIFT_Deleterious_Count'] = m.assign(z=m.SIFT_pred.astype(str).str.contains('D', na=False)).groupby('Gene').z.sum().reindex(out.Gene).values
            out['PolyPhen_Damaging_Count'] = m.assign(z=m.Polyphen2_HDIV_pred.astype(str).str.contains('D|P', regex=True, na=False)).groupby('Gene').z.sum().reindex(out.Gene).values
            out['Deleterious_Impact_Ratio'] = (out.SIFT_Deleterious_Count + out.PolyPhen_Damaging_Count).clip(upper=out.SNV_Count) / (2 * den)
            return (out, {'maf_rows': len(m), 'maf_samples': m.Tumor_Sample_Barcode.nunique(), 'NCBI_builds': sorted(m.NCBI_Build.dropna().unique().tolist())})

        def clinvar(target):
            x = pd.read_csv(CLIN, sep='\t', low_memory=False)
            x = x[x.GeneSymbol.astype(str).str.upper().isin(target)].copy()
            x['Gene'] = x.GeneSymbol.astype(str).str.upper()
            sig = x.ClinicalSignificance.astype(str).str.lower()
            typ = x.Type.astype(str).str.lower()
            rev = x.ReviewStatus.astype(str).str.lower()
            g = x.groupby('Gene')
            o = pd.DataFrame({'Gene': g.size().index, 'ClinVar_Total_Variants': g.size().values})
            for n, s in [('ClinVar_Pathogenic_Count', sig.str.contains('pathogenic')), ('ClinVar_Benign_Count', sig.str.contains('benign')), ('ClinVar_Uncertain_Count', sig.str.contains('uncertain')), ('ClinVar_Reviewed_Pathogenic', rev.str.contains('expert|guideline', regex=True)), ('ClinVar_SNV_Count', typ.str.contains('single nucleotide')), ('ClinVar_Indel_Count', typ.str.contains('deletion|insertion')), ('ClinVar_Structural_Count', typ.str.contains('duplication|copy|structural', regex=True))]:
                o[n] = x.assign(z=s).groupby('Gene').z.sum().reindex(o.Gene).values
            o['ClinVar_Pathogenic_Ratio'] = o.ClinVar_Pathogenic_Count / o.ClinVar_Total_Variants
            return o

        def pathways(target):
            d = defaultdict(set)
            for z in open(PATHWAYS, encoding='utf-8'):
                a = z.rstrip().split('\t')
                for g in a[2:]:
                    if g.upper() in target:
                        d[g.upper()].add(a[0])
            return pd.DataFrame({'Gene': list(d), 'pathway_count_all': [len(d[x]) for x in d]})

        def engineered(x):
            for c in ['SNV_Count', 'Missense', 'Nonsense', 'Frameshift', 'Silent', 'Splice_Site', 'Patients_Affected', 'ClinVar_Total_Variants', 'ClinVar_Pathogenic_Count', 'ClinVar_Benign_Count', 'ClinVar_Uncertain_Count', 'ClinVar_Reviewed_Pathogenic', 'ClinVar_SNV_Count', 'ClinVar_Indel_Count', 'ClinVar_Structural_Count', 'PPI_degree', 'CGC_neighbor_count', 'pathway_count_all', 'PPI_graph_degree', 'PPI_graph_weighted_degree', 'PPI_graph_avg_neighbor_degree', 'PPI_graph_max_neighbor_degree', 'PPI_graph_CGC_neighbor_count', 'PPI_graph_twohop_CGC_count', 'Prom_Meth_Probe_Count', 'SIFT_Deleterious_Count', 'PolyPhen_Damaging_Count']:
                if c in x:
                    x['FE_log1p_' + c] = np.log1p(x[c].clip(lower=0))
                    x['FE_sqrt_' + c] = np.sqrt(x[c].clip(lower=0))
            ns = x.Missense + x.Nonsense + x.Frameshift + x.Splice_Site
            tr = x.Nonsense + x.Frameshift + x.Splice_Site
            p = x.Patients_Affected
            for k, v in {'FE_mutations_per_patient': x.SNV_Count / (p + 1), 'FE_non_silent_count': ns, 'FE_non_silent_per_patient': ns / (p + 1), 'FE_truncating_count': tr, 'FE_truncating_per_patient': tr / (p + 1), 'FE_truncating_to_missense': tr / (x.Missense + 1), 'FE_high_impact_mutation_load': x.High_Impact_Ratio * np.log1p(x.SNV_Count), 'FE_pathogenicity_mutation_load': x.Pathogenicity_Score * np.log1p(ns), 'FE_network_total_signal': np.log1p(x.PPI_degree + x.PPI_graph_degree + x.PPI_graph_weighted_degree + x.CGC_neighbor_count + x.PPI_graph_CGC_neighbor_count + x.pathway_count_all), 'FE_graph_cgc_density': x.PPI_graph_CGC_neighbor_count / (x.PPI_graph_degree + 1), 'FE_graph_twohop_density': x.PPI_graph_twohop_CGC_count / (x.PPI_graph_degree + 1), 'FE_graph_broker_score': np.log1p(x.PPI_graph_degree) * x.PPI_graph_betweenness_approx, 'FE_graph_module_score': np.log1p(x.PPI_graph_CGC_neighbor_count + x.PPI_graph_twohop_CGC_count) * (1 - x.PPI_graph_clustering.clip(0, 1)), 'FE_pathway_graph_load': np.log1p(x.pathway_count_all) * np.log1p(x.PPI_graph_degree), 'FE_cgc_pathway_bridge': np.log1p(x.CGC_neighbor_count + x.PPI_graph_CGC_neighbor_count) * np.log1p(x.pathway_count_all), 'FE_weighted_degree_density': x.PPI_graph_weighted_degree / (x.PPI_graph_degree + 1), 'FE_clinvar_graph_interaction': np.log1p(x.ClinVar_Pathogenic_Count) * np.log1p(x.PPI_graph_CGC_neighbor_count), 'FE_cnv_gain_loss_delta': x.CNV_Gain_Ratio - x.CNV_Loss_Ratio, 'FE_cnv_amp_del_delta': x.CNV_Amplification_Ratio - x.CNV_Deletion_Ratio, 'FE_cnv_instability': x.CNV_Gain_Ratio + x.CNV_Loss_Ratio + x.CNV_Amplification_Ratio + x.CNV_Deletion_Ratio, 'FE_cnv_expression_load': x.CNV_Mean.abs() * x.Expr_CV.abs(), 'FE_methylation_delta': x.Meth_Hyper_Ratio - x.Meth_Hypo_Ratio, 'FE_promoter_methylation_confidence': x.Prom_Meth_Mean * np.log1p(x.Prom_Meth_Probe_Count), 'FE_expr_meth_interaction': x.Expr_CV * (x.Meth_Hyper_Ratio - x.Meth_Hypo_Ratio), 'FE_gc_entropy_balance': x.GC_Content * x.Entropy_2mer, 'FE_cpg_promoter_meth': x.CpG_Density * x.Prom_Meth_Mean, 'FE_clinvar_pathogenic_burden': x.ClinVar_Pathogenic_Count / (x.ClinVar_Total_Variants + 1), 'FE_clinvar_pathogenic_net': (x.ClinVar_Pathogenic_Count - x.ClinVar_Benign_Count) / (x.ClinVar_Total_Variants + 1), 'FE_clinvar_reviewed_pathogenic_burden': x.ClinVar_Reviewed_Pathogenic / (x.ClinVar_Total_Variants + 1)}.items():
                x[k] = v
            return x

        def locked(cohort):
            if cohort == 'CCRCC':
                return set(pd.read_csv(QC / 'ccrcc_gdc_selected_one_aliquot_per_case.csv').GDC_Case_Submitter_ID.astype(str))
            x = pd.read_csv(QC / 'hnscc_probe_level_vs_linkedomics_sample_overlap.csv')
            return set(x.loc[x.N_Data_Sources.eq(5), 'Sample_ID'].astype(str))

        def main():
            target = genes()
            mp = ensembl_map()
            base = seq_features(target)
            refs = {'fasta': str(FASTA), 'fasta_sha256': sha(FASTA), 'clinvar': str(CLIN), 'clinvar_sha256': sha(CLIN), 'pathway': str(PATHWAYS), 'pathway_sha256': sha(PATHWAYS), 'hgnc': str(HGNC), 'hgnc_sha256': sha(HGNC)}
            summary = {}
            for cohort in ['CCRCC', 'HNSCC']:
                lk = locked(cohort)
                e, c, sm = expression_cnv(cohort, lk, mp)
                mu, mm = mutation(cohort, lk)
                me = pd.read_csv(METH / (('ccrcc_95_selected' if cohort == 'CCRCC' else 'hnscc_102_complete_match') + '_gene_level_methylation_features.csv'))
                net = pd.read_csv(OUT / 'string_v12_network_features_4420.csv')
                parts = [base, e, c, mu, clinvar(set(target)), pathways(set(target)), me, net]
                x = pd.DataFrame({'Gene': target})
                for p in parts:
                    x = x.merge(p, on='Gene', how='left')
                x = engineered(x)
                feats = pd.read_csv(MODEL).Feature.tolist()
                x = x.reindex(columns=['Gene'] + feats)
                x.to_csv(OUT / f'{cohort.lower()}_locked_gene_by_305_model_features.csv', index=False)
                summary[cohort] = {'locked_samples': len(lk), **sm, **mm, 'genes': len(x), 'nonmissing_by_feature': x.drop(columns='Gene').notna().sum().to_dict()}
            pd.DataFrame([{'feature': f, 'CCRCC_nonmissing': summary['CCRCC']['nonmissing_by_feature'].get(f, 0), 'HNSCC_nonmissing': summary['HNSCC']['nonmissing_by_feature'].get(f, 0)} for f in pd.read_csv(MODEL).Feature]).to_csv(OUT / 'per_feature_missingness.csv', index=False)
            json.dump({'summary': summary, 'references': refs}, open(OUT / 'summary.json', 'w'), indent=2)
        if __name__ == '__main__':
            main()
    finally:
        _sys.argv = _runner_previous_argv


def main():
    parser = argparse.ArgumentParser(description='Download CPTAC inputs and build cohort feature tables.')
    parser.add_argument('task', choices=['download', 'build'])
    arguments = _sys.argv[1:]
    if not arguments or arguments[0] in {'-h', '--help'}:
        parser.print_help()
        return
    args = parser.parse_args(arguments[:1])
    forwarded = arguments[1:]
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if forwarded and args.task != 'download':
        parser.error('build does not accept extra arguments')
    {'download': _run_download, 'build': _run_build}[args.task](forwarded)

if __name__ == '__main__':
    main()
