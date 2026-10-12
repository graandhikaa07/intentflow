import win32com.client
from pathlib import Path

ppt_path = Path('artifacts/reports/FINAL_PROJECT_PRESENTATION.pptx').resolve()
pdf_path = Path('artifacts/reports/FINAL_PROJECT_PRESENTATION.pdf').resolve()
img_dir = Path('scratch/presentation_slides').resolve()
img_dir.mkdir(parents=True, exist_ok=True)

ppt_app = win32com.client.Dispatch('PowerPoint.Application')
pres = ppt_app.Presentations.Open(str(ppt_path), ReadOnly=1, Untitled=0, WithWindow=1)

# Export PDF
pres.SaveAs(str(pdf_path), 32) # ppSaveAsPDF
print(f"[SUCCESS] Exported PDF presentation to: {pdf_path}")

# Export slide PNGs
pres.Export(str(img_dir), 'PNG', 1920, 1080)
print(f"[SUCCESS] Exported slide PNGs to: {img_dir}")

slide_count = pres.Slides.Count
print(f"Total slides: {slide_count}")

pres.Close()
ppt_app.Quit()
