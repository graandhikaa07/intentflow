import json
import sys
from pathlib import Path

with open('scratch/paper_inventory.json', encoding='utf-8') as f:
    papers = json.load(f)

out = []
for p in papers:
    pid = p['id']
    folder = p['folder']
    fname = p['filename']
    pgs = p['pages']
    
    p1 = p.get('page1_text', '')
    p2 = p.get('page2_text', '')
    end = p.get('end_text', '')
    
    out.append(f"================================================================================")
    out.append(f"PAPER ID: {pid}")
    out.append(f"FOLDER: {folder}")
    out.append(f"FILENAME: {fname}")
    out.append(f"PAGES: {pgs}")
    out.append(f"--- PAGE 1 FIRST 1500 CHARS ---")
    out.append(p1[:1500])
    out.append(f"--- PAGE 2 FIRST 1000 CHARS ---")
    out.append(p2[:1000])
    out.append(f"--- END TEXT LAST 1000 CHARS ---")
    out.append(end[-1000:] if end else '')
    out.append("\n\n")

Path('scratch/all_papers_analysis.txt').write_text('\n'.join(out), encoding='utf-8')
print("Wrote all_papers_analysis.txt successfully.")
