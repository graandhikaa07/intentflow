import json
import sys
import re
from pathlib import Path

with open('scratch/paper_inventory.json', encoding='utf-8') as f:
    papers = json.load(f)

extracted = []

for p in papers:
    lines = [l.strip() for l in p['page1_text'].split('\n') if l.strip()]
    end_lines = [l.strip() for l in p.get('end_text', '').split('\n') if l.strip()]
    
    pid = p['id']
    folder = p['folder']
    fname = p['filename']
    
    extracted.append({
        'id': pid,
        'folder': folder,
        'filename': fname,
        'pages': p['pages'],
        'header_lines': lines[:20],
        'page2_lines': [l.strip() for l in p.get('page2_text', '').split('\n') if l.strip()][:15],
        'end_lines': end_lines[-30:] if end_lines else []
    })

with open('scratch/papers_extracted_headers.json', 'w', encoding='utf-8') as f:
    json.dump(extracted, f, indent=2, ensure_ascii=False)

print(f"Extracted headers for {len(extracted)} papers.")
