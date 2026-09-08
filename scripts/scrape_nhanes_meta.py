import urllib.request
import re
import os
import sys

def find_files():
    base_url = "https://wwwn.cdc.gov"
    results = {}
    
    for cycle, yr in [('G', '2011'), ('H', '2013')]:
        results[cycle] = {}
        for comp in ['Demographics', 'Laboratory', 'Examination', 'Questionnaire']:
            url = f"https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component={comp}&CycleBeginYear={yr}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
                # find matches for .XPT or .htm doc
                # Pattern: href="...XPT"
                matches = re.findall(r'href=[\'"]([^\'"]+\.XPT)[\'"]', html, re.IGNORECASE)
                results[cycle][comp] = matches
                print(f"Cycle {yr}-{int(yr)+1} ({cycle}) - {comp}: {len(matches)} XPT files")
            except Exception as e:
                print(f"Error {cycle} {comp}: {e}")
                
    return results

if __name__ == "__main__":
    res = find_files()
    for cycle in res:
        print(f"\n=== CYCLE {cycle} ===")
        for comp, links in res[cycle].items():
            print(f"-- {comp} --")
            for link in links:
                fname = os.path.basename(link)
                # check if interesting
                targets = ['GLU', 'GHB', 'DEMO', 'BMX', 'BPX', 'DIQ', 'PAQ', 'SLQ', 'PAX', 'PAM', 'CHOL', 'TRIG', 'HDL', 'INS']
                if any(t in fname.upper() for t in targets):
                    print(f"  {fname:15} -> {link}")
