"""Verify every DOI in paper_bspc/refs.bib against Crossref / DataCite (doi.org content negotiation). Read-only."""
import re, json, urllib.request, pathlib, sys, time
bib = pathlib.Path(sys.argv[1]).read_text()
out = []
for k, body in re.findall(r'@\w+\{([^,]+),(.*?)\n(?=@|\Z)', bib, re.S):
    m = re.search(r'doi\s*=\s*\{([^}]+)\}', body, re.I)
    t = re.search(r'\btitle\s*=\s*\{(.*)', body); bt = re.sub(r'[{}\\]', '', t.group(1))[:90] if t else ''
    y = re.search(r'year\s*=\s*\{?(\d{4})', body); by = y.group(1) if y else ''
    if not m: out.append(f"NODOI | {k} | {by} | {bt}"); continue
    doi = m.group(1).strip()
    try:
        rq = urllib.request.Request("https://doi.org/" + doi, headers={"Accept": "application/vnd.citationstyles.csl+json", "User-Agent": "ref-check (mailto:2612108@student.hcmute.edu.vn)"})
        j = json.loads(urllib.request.urlopen(rq, timeout=30).read().decode())
        ti = j.get("title", ""); ti = ti[0] if isinstance(ti, list) else ti
        ct = j.get("container-title", ""); ct = ct[0] if isinstance(ct, list) and ct else ct
        yr = (j.get("issued", {}).get("date-parts", [[None]])[0] or [None])[0]
        au = (j.get("author") or [{}])[0].get("family", "")
        out.append(f"OK | {k} | bib {by} | {bt}\n     -> {yr} | {au} | {str(ti)[:90]} | {str(ct)[:45]} | vol {j.get('volume','')} p {j.get('page', j.get('article-number',''))}")
    except Exception as e:
        out.append(f"FAIL | {k} | {doi} | {e}")
    time.sleep(0.3)
pathlib.Path(sys.argv[2]).write_text("\n".join(out)); print(len(out), "entries")
