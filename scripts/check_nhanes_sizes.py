import urllib.request
import os

files_to_check = {
    "DEMO_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DEMO_G.xpt",
    "GLU_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GLU_G.xpt",
    "GHB_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/GHB_G.xpt",
    "HDL_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/HDL_G.xpt",
    "TRIGLY_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/TRIGLY_G.xpt",
    "TCHOL_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/TCHOL_G.xpt",
    "BMX_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BMX_G.xpt",
    "BPX_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/BPX_G.xpt",
    "DIQ_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/DIQ_G.xpt",
    "PAQ_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAQ_G.xpt",
    "SLQ_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/SLQ_G.xpt",
    "PAXDAY_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXDAY_G.xpt",
    "PAXHR_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXHR_G.xpt",
    "PAXHD_G": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/PAXHD_G.xpt",
    "PAXMIN_G": "https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_G.xpt",
    
    # 2013-2014
    "DEMO_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
    "GLU_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GLU_H.xpt",
    "INS_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/INS_H.xpt",
    "GHB_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/GHB_H.xpt",
    "HDL_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/HDL_H.xpt",
    "TRIGLY_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/TRIGLY_H.xpt",
    "TCHOL_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/TCHOL_H.xpt",
    "BMX_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.xpt",
    "BPX_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.xpt",
    "DIQ_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DIQ_H.xpt",
    "PAQ_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAQ_H.xpt",
    "SLQ_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/SLQ_H.xpt",
    "PAXDAY_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXDAY_H.xpt",
    "PAXHR_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHR_H.xpt",
    "PAXHD_H": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/PAXHD_H.xpt",
    "PAXMIN_H": "https://ftp.cdc.gov/pub/NHANES/LargeDataFiles/PAXMIN_H.xpt",
}

for name, url in files_to_check.items():
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            size_mb = int(resp.headers.get('Content-Length', 0)) / (1024 * 1024)
            print(f"{name:12}: {size_mb:8.2f} MB | {url}")
    except Exception as e:
        print(f"{name:12}: ERROR - {e}")
