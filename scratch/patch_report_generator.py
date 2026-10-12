import shutil
import re
from pathlib import Path

orig_path = Path("scratch/generate_master_report.py")
backup_path = Path("scratch/generate_master_report.py.bak")

# Create backup
shutil.copyfile(orig_path, backup_path)
print(f"Backed up {orig_path} to {backup_path}")

content = orig_path.read_text(encoding='utf-8')

# 1. Update imports
import_old = """from report_builder_helpers import (
    set_cell_background, set_cell_margins, set_cell_border, add_callout
)"""

import_new = """from report_builder_helpers import (
    set_cell_background, set_cell_margins, set_cell_border, add_callout
)
from literature_review_builder import (
    add_literature_review_section, add_comprehensive_references_section
)"""

assert import_old in content, "Could not find import block"
content = content.replace(import_old, import_new, 1)

# 2. Add Literature Review section call right after Section 1
intro_end = """    add_p(
        "This research project proposes and evaluates IntentFlow, an intent-aware routing architecture. The foundational hypothesis is that "
        "by introducing a lightweight, highly accurate sequence classification model prior to generation, incoming queries can be automatically "
        "categorized by operational intent and dispatched to specialized execution handlers. Queries requiring deterministic computation are "
        "handled by local symbolic engines; factual queries are grounded against a verified reference corpus; structural formatting tasks are "
        "synthesized via deterministic schema engines; and only open-ended semantic adaptation prompts are delegated to a cloud foundation LLM."
    )"""

lit_review_insertion = """    add_p(
        "This research project proposes and evaluates IntentFlow, an intent-aware routing architecture. The foundational hypothesis is that "
        "by introducing a lightweight, highly accurate sequence classification model prior to generation, incoming queries can be automatically "
        "categorized by operational intent and dispatched to specialized execution handlers. Queries requiring deterministic computation are "
        "handled by local symbolic engines [7]; factual queries are grounded against a verified reference corpus [3], [22]; structural formatting tasks are "
        "synthesized via deterministic schema engines [5]; and only open-ended semantic adaptation prompts are delegated to a cloud foundation LLM [8]."
    )

    # =========================================================================
    # 2. LITERATURE REVIEW
    # =========================================================================
    add_literature_review_section(doc, add_h1, add_h2, add_h3, add_p, format_table, add_callout)"""

assert intro_end in content, "Could not find intro end block"
content = content.replace(intro_end, lit_review_insertion, 1)

# 3. Renumber sections 2 through 27 to 3 through 28
renumberings = [
    ('add_h1("2. Problem Statement")', 'add_h1("3. Problem Statement")'),
    ('add_h1("3. Research Objective")', 'add_h1("4. Research Objective")'),
    ('add_h1("4. Research Questions")', 'add_h1("5. Research Questions")'),
    ('add_h1("5. System Overview")', 'add_h1("6. System Overview")'),
    ('add_h1("6. System Architecture")', 'add_h1("7. System Architecture")'),
    ('add_h1("7. Intent Taxonomy")', 'add_h1("8. Intent Taxonomy")'),
    ('add_h1("8. Dataset Construction")', 'add_h1("9. Dataset Construction")'),
    ('add_h1("9. Benchmark Dataset & Integrity Audits")', 'add_h1("10. Benchmark Dataset & Integrity Audits")'),
    ('add_h1("10. Intent Classification Model")', 'add_h1("11. Intent Classification Model")'),
    ('add_h1("11. Classifier Training Results")', 'add_h1("12. Classifier Training Results")'),
    ('add_h1("12. Specialized Handlers")', 'add_h1("13. Specialized Handlers")'),
    ('add_h1("13. Routing Architecture")', 'add_h1("14. Routing Architecture")'),
    ('add_h1("14. Experimental Design")', 'add_h1("15. Experimental Design")'),
    ('add_h1("15. Evaluation Metrics")', 'add_h1("16. Evaluation Metrics")'),
    ('add_h1("16. Final Experiment Results")', 'add_h1("17. Final Experiment Results")'),
    ('add_h1("17. Per-Intent Analysis")', 'add_h1("18. Per-Intent Analysis")'),
    ('add_h1("18. Failure Analysis")', 'add_h1("19. Failure Analysis")'),
    ('add_h1("19. Matched-Pair Analysis")', 'add_h1("20. Matched-Pair Analysis")'),
    ('add_h1("20. Efficiency Analysis")', 'add_h1("21. Efficiency Analysis")'),
    ('add_h1("21. Cost–Accuracy–Latency Tradeoff")', 'add_h1("22. Cost–Accuracy–Latency Tradeoff")'),
    ('add_h1("22. Development and Boundary Evolution")', 'add_h1("23. Development and Boundary Evolution")'),
    ('add_h1("23. Limitations")', 'add_h1("24. Limitations")'),
    ('add_h1("24. Threats to Validity")', 'add_h1("25. Threats to Validity")'),
    ('add_h1("25. Practical Implications")', 'add_h1("26. Practical Implications")'),
    ('add_h1("26. Future Work")', 'add_h1("27. Future Work")'),
    ('add_h1("27. Conclusion")', 'add_h1("28. Conclusion")')
]

for old_h, new_h in renumberings:
    assert old_h in content, f"Could not find {old_h}"
    content = content.replace(old_h, new_h, 1)

# Also update comment dividers
for i in range(27, 1, -1):
    old_c = f"# {i}."
    new_c = f"# {i+1}."
    content = content.replace(old_c, new_c)

# 4. Replace placeholder and old references with comprehensive references
placeholder_old = """    # =========================================================================
    # PLACEHOLDER: LITERATURE REVIEW
    # =========================================================================
    doc.add_page_break()
    add_h1("Literature Review — To Be Added Separately")
    add_callout(
        doc,
        "LITERATURE REVIEW — TO BE ADDED SEPARATELY\\n\\n"
        "This master technical report contains all empirical data, methodology, architectural specifications, results, tables, and figures. "
        "The formal literature review (covering hallucination taxonomies, mixture-of-experts, semantic routing, and foundation model economics) "
        "will be prepared and integrated separately for the final academic paper submission. No literature references have been fabricated.",
        title="Literature Review Section Status",
        border_color="475569", bg_color="F1F5F9"
    )

    # =========================================================================
    # REFERENCES / DOCUMENTED SOURCE CORPUS
    # =========================================================================
    add_h1("Documented Source Corpus & Dataset Citations")
    add_p("The following verified dataset sources, software libraries, and foundation models were utilized in this project:")
    add_p(
        "1. Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K dataset).\\n"
        "2. Larson, S., et al. (2019). An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (CLINC150 corpus).\\n"
        "3. Joshi, M., et al. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.\\n"
        "4. Narayan, S., et al. (2018). Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization (XSum).\\n"
        "5. Zhou, J., et al. (2023). Instruction-Following Evaluation for Large Language Models (IFEval benchmark).\\n"
        "6. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv:1910.01108.\\n"
        "7. Meurer, A., et al. (2017). SymPy: symbolic computing in Python. PeerJ Computer Science, 3:e103.\\n"
        "8. Google GenAI SDK. (2025). Gemini 3.1 Flash-Lite Foundation Model API. Google Cloud Platform.\\n"
        "9. Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020."
    )"""

placeholder_new = """    # =========================================================================
    # REFERENCES
    # =========================================================================
    doc.add_page_break()
    add_comprehensive_references_section(doc, add_h1, add_p)"""

assert placeholder_old in content, "Could not find placeholder/references block"
content = content.replace(placeholder_old, placeholder_new, 1)

orig_path.write_text(content, encoding='utf-8')
print("Successfully patched scratch/generate_master_report.py!")
