import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/paper_inventory.json', encoding='utf-8') as f:
    papers = json.load(f)

for p in papers:
    print("=" * 80)
    print(f"ID {p['id']}: [{p['folder']}] {p['filename']} ({p['pages']} pages)")
    lines = [line.strip() for line in p['page1_text'].split('\n') if line.strip()]
    # Print the first 25 non-empty lines
    for line in lines[:25]:
        print("  ", line)
