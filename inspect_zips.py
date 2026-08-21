import zipfile
import glob
import os

zip_candidates = glob.glob(r"C:\Users\HP\Downloads\*.zip")
for zpath in zip_candidates:
    try:
        with zipfile.ZipFile(zpath, 'r') as z:
            names = z.namelist()
            # If relevant to d1namo, ecg, glucose, summary, etc.
            is_relevant = any(k in zpath.lower() for k in ['summary', 'archive', 'diabetes', 'files', 'd1namo']) or \
                          any(k in n.lower() for k in ['glucose', 'summary', 'ecg', 'sensor', 'breathing', 'accel'])
            if is_relevant:
                print(f"\n==========================================")
                print(f"ZIP: {zpath} (Size: {os.path.getsize(zpath)/(1024*1024):.2f} MB, Total Files: {len(names)})")
                print(f"Sample files (up to 15):")
                for n in names[:15]:
                    print(f"  - {n}")
                if len(names) > 15:
                    print(f"  ... and {len(names)-15} more files")
    except Exception as e:
        pass
