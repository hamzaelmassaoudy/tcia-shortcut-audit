"""RQ5 literature search (plan 7e). Runs every fixed query, saves all records and a search log.
Usage: python rq5_search.py   (writes records_raw.csv and search_log.md next to this file)
Each query is cached in raw/ when it succeeds, so a rerun repeats only the queries that failed."""
import csv, json, time, datetime, re
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

HERE = Path(__file__).parent
RAW = HERE / 'raw'
RAW.mkdir(exist_ok=True)
S = requests.Session()
S.headers['User-Agent'] = 'acquisition-shortcut-audit RQ5 search (research; contact via project)'
TODAY = datetime.date.today().isoformat()

def get(url, **kw):
    """GET with retries (connections from this computer drop now and then; APIs limit request rates)."""
    for wait in (0, 5, 15, 45, 90):
        time.sleep(wait)
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
    raise RuntimeError(f'failed after retries: {url}')

# --- fixed queries (plan 7e) -------------------------------------------------------------------------------
IMG = '("CT" OR "computed tomography")'
ARCH = '("Cancer Imaging Archive" OR "TCIA")'
EPMC = {
    'lung_A': f'("TCGA-LUAD" OR "TCGA LUAD") AND ("TCGA-LUSC" OR "TCGA LUSC") AND {ARCH} AND {IMG}',
    'lung_B': f'"adenocarcinoma" AND "squamous cell carcinoma" AND {ARCH} AND {IMG} AND '
              f'("radiomics" OR "deep learning" OR "machine learning" OR "convolutional neural network")',
    'kidney_A': f'("TCGA-KIRC" OR "TCGA KIRC" OR "TCGA-KIRP" OR "TCGA KIRP" OR "TCGA-KICH" OR "TCGA KICH") AND {ARCH} AND {IMG}',
    'kidney_B': f'"renal cell carcinoma" AND ("subtype" OR "subtypes" OR "papillary" OR "chromophobe") AND {ARCH} AND {IMG} '
                f'AND ("radiomics" OR "deep learning" OR "machine learning" OR "convolutional neural network")',
    'pda_A': '("CPTAC-PDA" OR "CPTAC PDA")',
    'lidc_A': '("LIDC-IDRI" OR "LIDC") AND ("diagnosis data" OR "tcia-diagnosis-data" OR "patient diagnosis" OR '
              '"157 patients" OR "pathologically confirmed" OR "pathologically proven" OR "biopsy-proven") AND '
              '("metastatic" OR "metastasis" OR "metastases")',
}
PUBMED = {
    'lung_P': '("adenocarcinoma"[tiab] AND "squamous"[tiab]) AND ("TCIA"[all] OR "Cancer Imaging Archive"[all] OR '
              '"TCGA"[tiab]) AND ("CT"[tiab] OR "computed tomography"[tiab]) AND ("radiomics"[tiab] OR '
              '"deep learning"[tiab] OR "machine learning"[tiab] OR "neural network"[tiab])',
    'kidney_P': '("renal cell carcinoma"[tiab]) AND ("subtype"[tiab] OR "subtypes"[tiab] OR "papillary"[tiab] OR '
                '"chromophobe"[tiab]) AND ("TCIA"[all] OR "Cancer Imaging Archive"[all] OR "TCGA"[tiab]) AND '
                '("CT"[tiab] OR "computed tomography"[tiab])',
    'pda_P': '"CPTAC"[tiab] AND ("pancreatic"[tiab]) AND ("CT"[tiab] OR "computed tomography"[tiab] OR "imaging"[tiab])',
    'lidc_P': '("LIDC"[tiab] OR "LIDC-IDRI"[tiab]) AND ("metastatic"[tiab] OR "metastasis"[tiab] OR "metastases"[tiab])',
}
ARXIV = {
    'lung_X': 'all:TCIA AND all:adenocarcinoma AND all:squamous',
    'lung_X2': 'all:"TCGA-LUAD"',
    'kidney_X': 'all:TCIA AND all:"renal cell carcinoma"',
    'kidney_X2': 'all:"TCGA-KIRC"',
    'pda_X': 'all:"CPTAC-PDA"',
    'lidc_X': 'all:LIDC AND all:metastatic',
}
S2 = {
    'lung_S': 'CT lung adenocarcinoma squamous cell carcinoma classification TCGA TCIA',
    'kidney_S': 'CT renal cell carcinoma subtype classification TCGA TCIA',
    'pda_S': 'CPTAC-PDA pancreatic CT',
    'lidc_S': 'LIDC-IDRI primary lung cancer versus metastatic classification diagnosis data',
}

def rec(source, qid, rid, title, abstract, year, venue, doi, url, extra=''):
    return {'source': source, 'query_id': qid, 'task': qid.split('_')[0], 'record_id': rid,
            'title': re.sub(r'\s+', ' ', title or '').strip(), 'abstract': re.sub(r'\s+', ' ', abstract or '').strip(),
            'year': year or '', 'venue': venue or '', 'doi': (doi or '').lower().strip(), 'url': url or '',
            'note': extra}

def europepmc(qid, q):
    out, cursor, total = [], '*', 0
    while True:
        r = get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',
                params={'query': q, 'format': 'json', 'pageSize': 1000, 'cursorMark': cursor, 'resultType': 'core'}).json()
        total = r.get('hitCount', 0)
        for it in r.get('resultList', {}).get('result', []):
            out.append(rec('Europe PMC', qid, it.get('id'), it.get('title'), it.get('abstractText'), it.get('pubYear'),
                           (it.get('journalInfo') or {}).get('journal', {}).get('title') or it.get('source'),
                           it.get('doi'), f"https://europepmc.org/article/{it.get('source')}/{it.get('id')}",
                           f"type={','.join((it.get('pubTypeList') or {}).get('pubType', []))}"))
        nxt = r.get('nextCursorMark')
        if not nxt or nxt == cursor or len(out) >= total:
            return total, out
        cursor = nxt

E = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
def pubmed(qid, q):
    for wait in (0, 10, 30, 60):              # NCBI sometimes answers with an error message instead of results
        time.sleep(wait)
        r = get(f'{E}/esearch.fcgi', params={'db': 'pubmed', 'term': q, 'retmax': 10000, 'retmode': 'json'}).json()
        if 'idlist' in r.get('esearchresult', {}):
            break
    ids, out = r['esearchresult']['idlist'], []
    for i in range(0, len(ids), 200):
        root = ET.fromstring(get(f'{E}/efetch.fcgi', params={'db': 'pubmed', 'id': ','.join(ids[i:i + 200]),
                                                              'retmode': 'xml'}).content)
        for art in root.findall('.//PubmedArticle'):
            pmid = art.findtext('.//PMID')
            t = art.find('.//ArticleTitle')
            out.append(rec('PubMed', qid, pmid, ''.join(t.itertext()) if t is not None else '',
                           ' '.join(''.join(a.itertext()) for a in art.findall('.//Abstract/AbstractText')),
                           art.findtext('.//JournalIssue/PubDate/Year') or art.findtext('.//ArticleDate/Year') or '',
                           art.findtext('.//Journal/Title') or '',
                           next((e.text for e in art.findall('.//ArticleId') if e.get('IdType') == 'doi'), ''),
                           f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/'))
    return int(r['esearchresult']['count']), out

NS = {'a': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
def arxiv(qid, q):
    root = ET.fromstring(get('https://export.arxiv.org/api/query', params={'search_query': q, 'max_results': 500}).content)
    out = []
    for e in root.findall('a:entry', NS):
        aid = e.findtext('a:id', '', NS).rsplit('/', 1)[-1]
        out.append(rec('arXiv', qid, aid, e.findtext('a:title', '', NS), e.findtext('a:summary', '', NS),
                       e.findtext('a:published', '', NS)[:4], 'arXiv', e.findtext('arxiv:doi', '', NS),
                       f'https://arxiv.org/abs/{aid}'))
    return len(out), out

def semantic_scholar(qid, q):
    data = get('https://api.semanticscholar.org/graph/v1/paper/search',
               params={'query': q, 'limit': 100, 'fields': 'title,abstract,year,venue,externalIds,url'}).json()
    out = []
    for p in data.get('data', []):
        ext = p.get('externalIds') or {}
        out.append(rec('Semantic Scholar', qid, p.get('paperId'), p.get('title'), p.get('abstract'), p.get('year'),
                       p.get('venue'), ext.get('DOI', ''), p.get('url'), f"arxiv={ext.get('ArXiv', '')}"))
    return data.get('total', len(out)), out      # relevance search: only the first 100 are kept

rows, log = [], []
for source, queries, fn in (('Europe PMC', EPMC, europepmc), ('PubMed', PUBMED, pubmed), ('arXiv', ARXIV, arxiv),
                            ('Semantic Scholar', S2, semantic_scholar)):
    for qid, q in queries.items():
        cache = RAW / f"{source.replace(' ', '_')}__{qid}.json"
        if cache.exists():
            c = json.loads(cache.read_text(encoding='utf-8'))
        else:
            try:
                total, out = fn(qid, q)
                c = {'date': TODAY, 'query': q, 'hits': total, 'records': out}
                cache.write_text(json.dumps(c), encoding='utf-8')
            except Exception as e:               # one unreachable source must not stop the others; logged
                c = {'date': TODAY, 'query': q, 'hits': f'FAILED: {e!r}'[:100], 'records': []}
            time.sleep(3)
        rows += c['records']
        log.append((source, qid, q, c['hits'], len(c['records']), c['date']))
        print(f"{source:<17} {qid:<10} hits {c['hits']} saved {len(c['records'])}", flush=True)

with open(HERE / 'records_raw.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(HERE / 'search_log.md', 'w', encoding='utf-8') as f:
    f.write('# RQ5 search log\n\nQueries of plan 7e, run with `rq5_search.py`. Hits = total reported by the source '
            '(Semantic Scholar is a relevance search: only its first 100 are kept); Saved = records kept.\n\n'
            '| Source | Query | Date | Hits | Saved | Query string |\n|---|---|---|---|---|---|\n')
    for src, qid, q, total, saved, date in log:
        f.write(f'| {src} | {qid} | {date} | {total} | {saved} | `{q}` |\n')
print(len(rows), 'records saved')
