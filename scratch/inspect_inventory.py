import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/paper_inventory.json', encoding='utf-8') as f:
    papers = json.load(f)

print(f"Total papers loaded: {len(papers)}")
for p in papers:
    print("=" * 80)
    pid = p['id']
    folder = p['folder']
    fname = p['filename']
    pgs = p['pages']
    print(f"ID {pid}: [{folder}] {fname} ({pgs} pages)")
    lines = [line.strip() for line in p['page1_text'].split('\n') if line.strip()]
    for l in lines[:10]:
        print("  ", l)
