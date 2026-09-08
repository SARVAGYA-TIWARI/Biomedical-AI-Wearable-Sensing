import os
import sys
import time
import urllib.request
import ssl

ssl_context = ssl._create_unverified_context()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'nhanes')
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    # ── 2011-2012 (Cycle G) ──────────────────────────────────────────────────
    "DEMO_G.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
    "GLU_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GLU_G.xpt",
    "GHB_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GHB_G.xpt",
    "HDL_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HDL_G.xpt",
    "TRIGLY_G.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/TRIGLY_G.xpt",
    "TCHOL_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/TCHOL_G.xpt",
    "BMX_G.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.xpt",
    "BPX_G.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPX_G.xpt",
    "DIQ_G.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.xpt",
    "PAQ_G.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAQ_G.xpt",
    "SLQ_G.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SLQ_G.xpt",
    "PAXHD_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXHD_G.xpt",
    "PAXDAY_G.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXDAY_G.xpt",
    "PAXHR_G.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXHR_G.xpt",

    # ── 2013-2014 (Cycle H) ──────────────────────────────────────────────────
    "DEMO_H.XPT":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
    "GLU_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GLU_H.xpt",
    "INS_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/INS_H.xpt",
    "GHB_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GHB_H.xpt",
    "HDL_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HDL_H.xpt",
    "TRIGLY_H.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/TRIGLY_H.xpt",
    "TCHOL_H.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/TCHOL_H.xpt",
    "BMX_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.xpt",
    "BPX_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.xpt",
    "DIQ_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.xpt",
    "PAQ_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.xpt",
    "SLQ_H.XPT":    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SLQ_H.xpt",
    "PAXHD_H.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHD_H.xpt",
    "PAXDAY_H.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXDAY_H.xpt",
    "PAXHR_H.XPT":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHR_H.xpt",
}

def download_file(filename, url):
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[EXISTS] {filename:14} ({size_mb:.2f} MB)", flush=True)
        return True

    print(f"[DOWNLOADING] {filename:14} from {url} ...", flush=True)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120, context=ssl_context) as resp, open(dest_path, 'wb') as out_f:
            chunk_size = 64 * 1024
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
        elapsed = time.time() - t0
        size_mb = downloaded / (1024 * 1024)
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"[COMPLETED]   {filename:14} ({size_mb:.2f} MB in {elapsed:.1f}s, {speed:.2f} MB/s)", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR]       {filename:14} Failed: {e}", flush=True)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    print("=" * 70, flush=True)
    print("NHANES 2011-2014 Comprehensive Automated Downloader", flush=True)
    print(f"Target Directory: {DATA_DIR}", flush=True)
    print(f"Total Files Scheduled: {len(FILES)}", flush=True)
    print("=" * 70, flush=True)

    success_count = 0
    start_total = time.time()
    for filename, url in FILES.items():
        if download_file(filename, url):
            success_count += 1

    total_time = time.time() - start_total
    print("=" * 70, flush=True)
    print(f"Download Summary: {success_count}/{len(FILES)} succeeded in {total_time:.1f}s", flush=True)
    total_size_mb = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in os.listdir(DATA_DIR) if f.endswith('.XPT')) / (1024 * 1024)
    print(f"Total NHANES Disk Usage: {total_size_mb:.2f} MB", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
