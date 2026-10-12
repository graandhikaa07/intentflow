"""
Full Master Technical Research Report Generator
Builds artifacts/reports/MASTER_TECHNICAL_RESEARCH_REPORT.docx
and exports artifacts/reports/MASTER_TECHNICAL_RESEARCH_REPORT.pdf
"""

import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import helpers
sys.path.insert(0, str(PROJECT_ROOT / "scratch"))
from report_builder_helpers import (
    set_cell_background, set_cell_margins, set_cell_border, add_callout
)
from literature_review_builder import (
    add_literature_review_section, add_comprehensive_references_section
)

def create_report():
    doc = docx.Document()
    
    # Page setup - Standard Letter, 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11.0)
        
        # Header / Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("IntentFlow: Technical Research Report | Cost-Accuracy Tradeoff Study")
        hrun.font.name = 'Calibri'
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Confidential Academic & Research Pre-Print — Master Project Documentation")
        frun.font.name = 'Calibri'
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

    # Styles setup
    styles = doc.styles
    normal_style = styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Calibri'
    normal_font.size = Pt(11)
    normal_font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # Helper styling functions
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0F, 0x29, 0x42) # Deep navy
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x78)
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
        return p

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.font.name = 'Calibri'
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        return p

    def add_fig_image(img_path, caption, width=Inches(6.2)):
        if os.path.exists(img_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run()
            run.add_picture(str(img_path), width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            p_cap.paragraph_format.keep_with_next = False
            run_cap = p_cap.add_run(caption)
            run_cap.font.name = 'Calibri'
            run_cap.font.size = Pt(9.5)
            run_cap.font.italic = True
            run_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        else:
            print(f"Warning: Image not found: {img_path}")

    def format_table(tbl, col_widths, headers, rows):
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.autofit = False
        
        # Header Row
        hdr_cells = tbl.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            hdr_cells[i].width = col_widths[i]
            set_cell_background(hdr_cells[i], "1F4E78")
            set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
            set_cell_border(hdr_cells[i], 
                            top=dict(val='single', sz=6, color='0F2942'),
                            bottom=dict(val='single', sz=12, color='0F2942'),
                            left=dict(val='none'), right=dict(val='none'))
            p = hdr_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = 'Calibri'
                run.font.size = Pt(9.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                
        # Body Rows
        for r_idx, row_data in enumerate(rows):
            row_cells = tbl.add_row().cells
            bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
            for c_idx, val in enumerate(row_data):
                row_cells[c_idx].text = str(val)
                row_cells[c_idx].width = col_widths[c_idx]
                set_cell_background(row_cells[c_idx], bg)
                set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=120, right=120)
                set_cell_border(row_cells[c_idx], 
                                top=dict(val='single', sz=4, color='E2E8F0'),
                                bottom=dict(val='single', sz=4, color='E2E8F0'),
                                left=dict(val='none'), right=dict(val='none'))
                p = row_cells[c_idx].paragraphs[0]
                # Center align if short numbers/percentages
                if any(k in headers[c_idx].lower() for k in ['%', 'count', 'total', 'f1', 'latency', 'accuracy', 'calls', 'share', 'cost', 'support']):
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    run.font.name = 'Calibri'
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                    
        p_space = doc.add_paragraph()
        p_space.paragraph_format.space_before = Pt(0)
        p_space.paragraph_format.space_after = Pt(8)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(36)
    title_p.paragraph_format.space_after = Pt(12)
    run_t = title_p.add_run("User Intent Classification for Hallucination Reduction in LLMs:\nA Cost-Accuracy Tradeoff Study")
    run_t.font.name = 'Calibri'
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x0F, 0x29, 0x42)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(24)
    run_sub = sub_p.add_run("Master Technical Research & Comprehensive Scientific Report\nNeural Networks and Deep Learning Research Project")
    run_sub.font.name = 'Calibri'
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    meta_tbl = doc.add_table(rows=6, cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Implementation:", "IntentFlow (Intent-Aware LLM Routing Architecture)"),
        ("Primary Research Focus:", "Pre-Generation Query Classification, Tool Dispatch, & Tradeoff Analysis"),
        ("Classification Engine:", "Fine-Tuned DistilBERT-base-uncased (66M Parameters, 4 Intent Classes)"),
        ("Baseline Foundation Model:", "Google Gemini 3.1 Flash-Lite (temperature=0.0 direct, temperature=0.2 routed)"),
        ("Benchmark Dataset:", "Controlled 80-Query Evaluation Benchmark (SHA-256 Verified)"),
        ("Report Classification:", "Complete Empirical Technical Record (Pre-Print Paper Material)")
    ]
    for i, (k, v) in enumerate(meta_data):
        c0, c1 = meta_tbl.rows[i].cells
        c0.width, c1.width = Inches(2.2), Inches(4.3)
        c0.text, c1.text = k, v
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        set_cell_border(c0, top=dict(val='single', sz=4, color='CBD5E1'), bottom=dict(val='single', sz=4, color='CBD5E1'), left=dict(val='none'), right=dict(val='none'))
        set_cell_border(c1, top=dict(val='single', sz=4, color='CBD5E1'), bottom=dict(val='single', sz=4, color='CBD5E1'), left=dict(val='none'), right=dict(val='none'))
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9.5)
        c1.paragraphs[0].runs[0].font.size = Pt(9.5)
    
    doc.add_page_break()

    # =========================================================================
    # ABSTRACT
    # =========================================================================
    add_h1("Abstract")
    add_p(
        "Monolithic Large Language Models (LLMs) are routinely deployed as universal interfaces for diverse inquiry "
        "distributions, spanning exact mathematical computation, closed-world factual retrieval, open-ended creative adaptation, "
        "and strict schema serialization. However, routing all queries through an unconstrained, autoregressive generation "
        "pathway incurs substantial financial costs, excessive end-to-end latency, and vulnerability to catastrophic hallucinations. "
        "This project investigates whether pre-generation user intent classification can reliably route incoming queries to specialized, "
        "domain-tailored execution handlers—combining deterministic symbolic engines, grounded reference corpus retrieval, and "
        "constrained schema solvers alongside cloud foundation models. We design and implement IntentFlow, a dual-mode routing "
        "framework driven by a lightweight, fine-tuned DistilBERT sequence classifier (66M parameters) mapped to a rigorous four-class "
        "taxonomy (MATHDS, TXTSRC, SEMADPT, CSTRSET). On a curated dataset of 26,492 examples (Dataset V3), the classifier achieves "
        "99.77% overall accuracy and 99.77% macro-averaged F1 on a held-out test split (N = 2,650). In a controlled, head-to-head "
        "benchmark across 80 challenging queries, the direct baseline (Gemini 3.1 Flash-Lite) achieved 100.00% benchmark correctness. "
        "The proposed intent-routed architecture achieved 96.25% response correctness (77/80 queries), with failures traced to an evaluator "
        "substring artifact and two creative/structural constraint boundary misclassifications. Crucially, the routed system achieved a "
        "67.13% reduction in total token usage (10,170 down to 3,343 tokens), a 68.15% reduction in API financial cost ($0.002232 down to "
        "$0.000711), a 77.50% reduction in external API invocations (offloading 62 of 80 queries entirely to local engines), and a 73.07% "
        "reduction in mean response latency (3,266.48 ms down to 879.63 ms), with median (P50) latency plummeting by 92.77% (1,921.85 ms "
        "down to 139.04 ms). Consequently, the empirical findings demonstrate a Pareto cost-accuracy-latency tradeoff: substantial "
        "reductions in latency, operational expenditure, and external cloud dependencies can be realized through intent routing, with a modest "
        "tradeoff in response correctness under complex constraint boundaries."
    )
    add_callout(
        doc,
        "Core Finding: The experimental results establish a multi-objective COST–ACCURACY–LATENCY TRADEOFF. Intent-aware routing "
        "slashes median latency by 92.77%, cuts token consumption by 67.13%, and offloads 77.50% of API calls to local deterministic handlers, "
        "while achieving 96.25% response correctness compared to the 100.00% score of the direct cloud LLM baseline.",
        title="Key Analytical Conclusion",
        border_color="0284C7", bg_color="F0F9FF"
    )

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    add_h1("1. Introduction")
    add_p(
        "Modern Large Language Models (LLMs), such as GPT-4, Claude 3.5, and Gemini 3.1, have redefined automated natural "
        "language understanding and generation. Due to their vast pre-training across trillions of tokens, these models are frequently "
        "treated as monolithic 'black-box' solvers capable of addressing any prompt submitted by an end user. In production environments, "
        "an identical autoregressive generation pipeline is typically invoked whether a user requests the solution to an algebraic equation, "
        "the historical date of a treaty, a creative sonnet, or a strict JSON-formatted payload."
    )
    add_p(
        "Despite their extraordinary versatility, routing all user queries indiscriminately through a large foundation model suffers from "
        "three fundamental architectural inefficiencies:"
    )
    add_p(
        "1. Vulnerability to Hallucination: Autoregressive models generate text probabilistically based on next-token distribution statistics. "
        "Consequently, they lack intrinsic guarantees of mathematical truth or closed-world factual grounding, leading to subtle arithmetic errors "
        "and confabulated factual claims.\n"
        "2. Resource and Latency Inefficiency: Invocations of remote cloud APIs require substantial network round-trip overhead and token decoding "
        "time. Even lightweight models exhibit latencies between 1.5 and 8.5 seconds per query, which is highly inefficient for deterministic operations "
        "that could execute locally in under 2 milliseconds.\n"
        "3. Financial and API Cost Accumulation: Enterprise systems face escalating token fees when relying exclusively on external commercial APIs, "
        "especially for high-throughput, repetitive tasks that do not genuinely require open-ended neural generation."
    )
    add_p(
        "This research project proposes and evaluates IntentFlow, an intent-aware routing architecture. The foundational hypothesis is that "
        "by introducing a lightweight, highly accurate sequence classification model prior to generation, incoming queries can be automatically "
        "categorized by operational intent and dispatched to specialized execution handlers. Queries requiring deterministic computation are "
        "handled by local symbolic engines [7]; factual queries are grounded against a verified reference corpus [3], [22]; structural formatting tasks are "
        "synthesized via deterministic schema engines [5]; and only open-ended semantic adaptation prompts are delegated to a cloud foundation LLM [8]."
    )

    # =========================================================================
    # 2. LITERATURE REVIEW
    # =========================================================================
    add_literature_review_section(doc, add_h1, add_h2, add_h3, add_p, format_table, add_callout)

    # =========================================================================
    # 3. PROBLEM STATEMENT
    # =========================================================================
    add_h1("3. Problem Statement")
    add_p(
        "The central engineering problem addressed in this study is the unguided dispatch problem in modern LLM systems. When a user interacts "
        "with an AI system, their query embodies an implicit computational contract. Consider four representative prompt archetypes:"
    )
    add_p(
        "• Prompt A (Mathematical Reasoning): 'Solve for x: 3x^2 - 12x + 9 = 0.'\n"
        "  Contract: Requires absolute numerical exactness. A probabilistic guess or rounding error constitutes a complete failure.\n"
        "• Prompt B (Factual Retrieval): 'Who was the first British woman to win a Nobel Prize?'\n"
        "  Contract: Requires strictly grounded historical evidence with verifiable attribution, free from unreferenced assertions.\n"
        "• Prompt C (Semantic Adaptation): 'Rewrite this memo in a professional and empathetic tone.'\n"
        "  Contract: Requires stylistic nuance, semantic fluency, and open-ended generative creativity.\n"
        "• Prompt D (Constraint Enforcement): 'Extract user attributes as a JSON object with keys: name (str), age (int), active (bool).'\n"
        "  Contract: Requires zero schema drift, strict syntax compliance, and complete omission of conversational filler or markdown fences."
    )
    add_p(
        "When an unguided system directs Prompts A, B, and D to a general-purpose LLM, it uses an over-parameterized, non-deterministic "
        "mechanism for problems that possess closed-form, deterministic solutions. Conversely, attempting to use static rule-based systems "
        "for natural language fails due to linguistic variability. The core technical challenge is therefore twofold: (1) Can a compact "
        "neural classifier accurately discriminate subtle intent boundaries at sub-50ms latencies? and (2) Does specialized routing yield "
        "a superior operational trade-off compared to state-of-the-art commercial LLMs?"
    )

    # =========================================================================
    # 4. RESEARCH OBJECTIVE
    # =========================================================================
    add_h1("4. Research Objective")
    add_p(
        "The overarching objective of this investigation is to empirically evaluate whether intent-based routing to specialized handlers "
        "can eliminate hallucinations in deterministic domains and improve operational efficiency compared to monolithic LLM generation. "
        "Specifically, the project investigates the following primary objective:"
    )
    add_callout(
        doc,
        "Primary Research Objective: Design, train, and empirically benchmark a multi-tier query routing architecture to determine "
        "whether pre-generation intent classification can simultaneously reduce factual/arithmetic failure rates, eliminate unnecessary "
        "cloud API consumption, and lower response latency relative to direct foundation LLM execution.",
        title="Formal Research Objective",
        border_color="1F4E78", bg_color="F8FAFC"
    )
    add_p(
        "As experimental findings accumulated across developmental phases, a crucial secondary objective emerged: characterizing the "
        "multi-dimensional Pareto frontier between response correctness, end-to-end latency, and monetary cost when local deterministic "
        "tools are substituted for foundation models."
    )

    # =========================================================================
    # 5. RESEARCH QUESTIONS
    # =========================================================================
    add_h1("5. Research Questions")
    add_p("To guide the empirical evaluation, the study formulates five core research questions (RQs):")
    add_p(
        "• RQ1 (Classification Fidelity): How accurately and reliably can a lightweight pre-trained transformer (DistilBERT, 66M parameters) "
        "discriminate user operational intent across mathematical, factual, creative, and constraint-oriented query distributions?\n"
        "• RQ2 (Comparative Correctness): How does an intent-routed pipeline compare against an unguided foundation LLM (Gemini 3.1 Flash-Lite) "
        "in raw response correctness across a rigorous, multi-domain evaluation benchmark?\n"
        "• RQ3 (Latency Profiling): What is the quantitative impact of intent-aware local dispatch on mean, median (P50), and tail (P95) "
        "latency profiles relative to remote API execution?\n"
        "• RQ4 (Resource & Cost Economics): To what extent does offloading deterministic queries to local execution handlers reduce token "
        "consumption, cloud API invocation frequency, and financial operating costs?\n"
        "• RQ5 (Failure Taxonomy & Boundary Sensitivity): What specific structural characteristics cause intent classification errors, and "
        "how do boundary misclassifications between semantic creativity and constraint enforcement impact end-to-end task success?"
    )

    # =========================================================================
    # 6. SYSTEM OVERVIEW
    # =========================================================================
    add_h1("6. System Overview")
    add_p(
        "The developed system, designated IntentFlow, implements a complete query processing pipeline that decouples intent comprehension "
        "from response generation. The architecture supports two distinct operational modes:"
    )
    add_h2("5.1 Canonical Auto-Detect Pipeline (Benchmark Mode)")
    add_p(
        "Under default operation, the system processes queries autonomously through a four-stage sequential pipeline:"
    )
    add_p(
        "1. Query Ingestion: The user query is captured in raw text format by the composer interface.\n"
        "2. Intent Inference: The query is tokenized and processed by a frozen, fine-tuned DistilBERT sequence classifier. The classifier outputs "
        "the predicted intent label alongside softmax probability confidence scores.\n"
        "3. Router Dispatch: The QueryRouter maps the predicted label to its designated specialized execution handler.\n"
        "4. Handler Execution & Response Aggregation: The target handler processes the query, records local execution metrics, and passes the "
        "result to the ResponseAggregator, which formats the user-facing text and constructs comprehensive technical telemetry."
    )
    add_h2("5.2 User-Provided Intent Mode (Interactive UI Utility)")
    add_p(
        "In the final interactive deployment, IntentFlow provides an optional user-selected intent dropdown alongside the query composer. "
        "If an end user explicitly designates a target intent (e.g., SEMADPT) prior to submission, the system treats this choice as authoritative. "
        "In this mode, DistilBERT classifier inference is completely bypassed (incurring 0.0 ms classification latency), avoiding redundant computation, "
        "and the query is dispatched directly to the selected handler. It is critical to note that this capability is an interactive application utility "
        "and was strictly excluded from the controlled Phase 6.8 scientific benchmark to maintain empirical integrity."
    )

    # =========================================================================
    # 7. SYSTEM ARCHITECTURE
    # =========================================================================
    add_h1("7. System Architecture")
    add_p(
        "Figure 1 illustrates the high-level system architecture of IntentFlow, highlighting the interplay between the query composer, "
        "the DistilBERT sequence classification head, the central QueryRouter, the four specialized execution handlers, and the unified "
        "ResponseAggregator."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig1_system_architecture.png", 
                  "Figure 1: High-level architectural topology of the IntentFlow intent-aware routing framework.")
    add_p(
        "Figure 2 outlines the end-to-end decision and execution flowchart, demonstrating how queries diverge between the canonical "
        "automatic classification pipeline and the user-provided intent bypass route."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig2_routing_flow.png", 
                  "Figure 2: End-to-end operational flow diagram detailing automated classification, handler dispatch, and telemetry generation.")

    # =========================================================================
    # 8. INTENT TAXONOMY
    # =========================================================================
    add_h1("8. Intent Taxonomy")
    add_p(
        "A foundational contribution of this research is the formulation of a four-class operational intent taxonomy. Unlike conversational "
        "dialogue taxonomies (e.g., booking flights or setting alarms), this taxonomy categorizes queries by their underlying computational "
        "execution requirements. Figure 3 illustrates the four taxonomy quadrants alongside their execution guarantees."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig3_intent_taxonomy.png", 
                  "Figure 3: Four-intent taxonomy matrix detailing architectural roles and execution guarantees.")
    
    add_p("Table 1 summarizes the formal definitions, typical prompt patterns, and assigned handlers across the taxonomy.")
    
    t1_headers = ["Intent Code", "Descriptive Name", "Typical Query Patterns", "Assigned Handler", "Execution Paradigm"]
    t1_widths = [Inches(1.0), Inches(1.3), Inches(2.0), Inches(1.2), Inches(1.0)]
    t1_rows = [
        ["MATHDS", "Mathematical Reasoning", "Arithmetic, polynomial roots, prime factorization, calculus, statistics", "MathSolverHandler", "Local SymPy (Deterministic)"],
        ["TXTSRC", "Factual Information", "Historical dates, biographical milestones, geography, scientific facts", "SearchRAGHandler", "Grounded Corpus (Citations)"],
        ["SEMADPT", "Semantic / Creative", "Creative stories, poems, dialogue synthesis, register/tone adaptation", "LLMGeneratorHandler", "Cloud Gemini API (Generative)"],
        ["CSTRSET", "Constraint Enforcement", "JSON schemas, XML trees, YAML blocks, delimited array bounds", "ConstrainedSolverHandler", "Local Schema (Synthesizer)"]
    ]
    t1 = doc.add_table(rows=1, cols=5)
    format_table(t1, t1_widths, t1_headers, t1_rows)

    add_h2("7.1 The SEMADPT vs. CSTRSET Boundary Dilemma")
    add_p(
        "A critical theoretical insight identified during system development was the structural tension between SEMADPT and CSTRSET. "
        "In naive taxonomy formulations, queries containing words like 'format', 'lines', or 'rules' are frequently conflated. For example, "
        "consider the query: 'Write a 4-line rhyming poem with an AABB scheme about autumn.' This prompt contains explicit constraints (4 lines, "
        "rhyme scheme), yet its core communicative purpose is aesthetic, open-ended natural language generation (SEMADPT). Conversely, 'Return a JSON "
        "list containing exactly 4 fruits' is a rigid serialization task (CSTRSET). Disentangling this boundary required specialized training interventions."
    )

    # =========================================================================
    # 9. DATASET CONSTRUCTION
    # =========================================================================
    add_h1("9. Dataset Construction")
    add_p(
        "To train a robust sequence classifier capable of high-precision routing, we constructed a comprehensive, multi-source dataset. "
        "The final training distribution, designated Dataset V3 (Boundary-Augmented), comprises exactly 26,492 curated records partitioned into "
        "train (80%), validation (10%), and held-out test (10%) splits. Figure 4 displays the class distribution and partition proportions."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig4_dataset_distribution.png", 
                  "Figure 4: Final dataset V3 class balance (left) and partition splits (right).")
    
    add_p("Table 2 documents the exact sample counts across dataset splits, while Table 3 details the empirical source provenance.")
    
    t2_headers = ["Intent Class", "Train Count (80%)", "Val Count (10%)", "Test Count (10%)", "Total V3 Count", "Class Proportion"]
    t2_widths = [Inches(1.2), Inches(1.1), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.1)]
    t2_rows = [
        ["MATHDS", "6,730", "841", "842", "8,413", "31.76%"],
        ["TXTSRC", "5,177", "647", "647", "6,471", "24.43%"],
        ["SEMADPT", "4,934", "617", "617", "6,168", "23.28%"],
        ["CSTRSET", "4,352", "544", "544", "5,440", "20.53%"],
        ["Total", "21,193", "2,649", "2,650", "26,492", "100.00%"]
    ]
    t2 = doc.add_table(rows=1, cols=6)
    format_table(t2, t2_widths, t2_headers, t2_rows)

    t3_headers = ["Source Corpus / Origin", "Primary Intent Focus", "Sample Count", "Dataset Share (%)"]
    t3_widths = [Inches(2.5), Inches(1.8), Inches(1.1), Inches(1.1)]
    t3_rows = [
        ["GSM8K (Grade School Math)", "MATHDS (Multi-step math)", "7,473", "28.21%"],
        ["CLINC150 Out-of-Scope Corpus", "TXTSRC, SEMADPT, CSTRSET", "6,996", "26.41%"],
        ["RECAST-30K Benchmark", "CSTRSET (Structural rule sets)", "3,000", "11.32%"],
        ["XSum (Extreme Summarization)", "SEMADPT (Semantic synthesis)", "2,750", "10.38%"],
        ["ExistingCustomGenerator", "Balanced synthetic queries", "1,511", "5.70%"],
        ["ExistingProjectData", "Historical verified prompts", "1,471", "5.55%"],
        ["TriviaQA Open-Domain Corpus", "TXTSRC (Fact retrieval)", "1,440", "5.44%"],
        ["BoundaryAugmentation_Phase6_7", "SEMADPT (400) / CSTRSET (400)", "800", "3.02%"],
        ["Google IFEval Benchmark", "CSTRSET (Instruction following)", "541", "2.04%"],
        ["RECAST-Test Suite", "CSTRSET (Verification rules)", "490", "1.85%"],
        ["BoundaryPatternDefense", "MATHDS, TXTSRC boundary anchors", "20", "0.08%"],
        ["Total Verified Records", "All Four Intent Classes", "26,492", "100.00%"]
    ]
    t3 = doc.add_table(rows=1, cols=4)
    format_table(t3, t3_widths, t3_headers, t3_rows)

    # =========================================================================
    # 10. BENCHMARK DATASET & INTEGRITY AUDITS
    # =========================================================================
    add_h1("10. Benchmark Dataset & Integrity Audits")
    add_p(
        "To rigorously evaluate end-to-end task completion without data snooping, the project maintained a strictly frozen evaluation "
        "suite located at data/hallucination_eval/eval_queries.json. The benchmark comprises exactly 80 queries, perfectly balanced with "
        "20 queries per intent class:"
    )
    add_p(
        "• MATHDS (MATH_01 to MATH_20): Arithmetic operations, quadratic and linear equations, percentage computations, prime tests, and statistical means.\n"
        "• TXTSRC (TXT_01 to TXT_20): Factual historical events, scientific constants, capital cities, and biographical achievements linked to reference passages.\n"
        "• SEMADPT (SEM_01 to SEM_20): Creative narratives, tone shifts, metaphors, constrained poems, and dialogue generation.\n"
        "• CSTRSET (CSTR_01 to CSTR_20): Complex JSON schema extraction, YAML configurations, XML structures, and strict delimited formats."
    )
    add_p(
        "For the factual evaluation, a companion corpus comprising 20 curated reference documents (reference_corpus.json, passages REF_01 through REF_20) "
        "was deployed. To prevent data contamination, exact-string and high-Jaccard token overlap audits were enforced before training. The benchmark "
        "files were monitored continuously via cryptographic hashing, maintaining identical SHA-256 signatures across all experimental phases:"
    )
    add_p(
        "• eval_queries.json SHA-256: 83ccec72b5a0c969e77e36eb00296e1c6e18f9369751038b80a829444f8ee454\n"
        "• reference_corpus.json SHA-256: 0b390fdb556ac1eecc3b51f37cdfa8d31549b068c824f340b2507e231d161178"
    )

    # =========================================================================
    # 11. INTENT CLASSIFICATION MODEL
    # =========================================================================
    add_h1("11. Intent Classification Model")
    add_p(
        "The core routing intelligence is driven by a fine-tuned DistilBERT-base-uncased transformer model (Sanwh et al., 2019). "
        "DistilBERT was selected because it retains 97% of BERT's language comprehension capabilities while utilizing 40% fewer parameters "
        "and running 60% faster, making it an ideal candidate for sub-50ms CPU routing in production pipelines. A standard sequence classification "
        "head (linear projection layer mapping the 768-dimensional pooled [CLS] embedding to 4 logits) was appended."
    )
    add_p("Table 4 outlines the complete hyperparameter and training environment configuration.")
    
    t4_headers = ["Hyperparameter / Setting", "Configured Value", "Operational Rationale"]
    t4_widths = [Inches(2.2), Inches(1.8), Inches(2.5)]
    t4_rows = [
        ["Base Pretrained Architecture", "distilbert-base-uncased", "66M parameters, 6 layers, 12 attention heads"],
        ["Target Classes", "4 (MATHDS, TXTSRC, SEMADPT, CSTRSET)", "Exact mapping to operational execution handlers"],
        ["Loss Function", "CrossEntropyLoss()", "Multi-class sequence classification objective"],
        ["Optimizer & Weight Decay", "AdamW (weight_decay = 0.01)", "Decay applied to weights; excluded from bias/LayerNorm"],
        ["Learning Rate & Schedule", "2e-5 (Linear Warmup + Decay)", "10% warmup steps (663 steps), linear decay to 0"],
        ["Batch Size & Padding", "32 (Dynamic Sequence Padding)", "Max sequence length 128 tokens; dynamic padding collator"],
        ["Training Duration", "4 Epochs (2,652 total steps)", "Early checkpointing based on held-out validation loss"],
        ["Hardware Environment", "12-Core Logical CPU Execution", "Threaded PyTorch CPU runtime (portable, edge-ready)"],
        ["Final Frozen Checkpoint", "artifacts/checkpoints/best_distilbert_final", "Model size: 267.8 MB (safetensors format)"]
    ]
    t4 = doc.add_table(rows=1, cols=3)
    format_table(t4, t4_widths, t4_headers, t4_rows)

    # =========================================================================
    # 12. CLASSIFIER TRAINING RESULTS
    # =========================================================================
    add_h1("12. Classifier Training Results")
    add_p(
        "The classifier converged rapidly during Phase 6.7 training, achieving a peak validation accuracy of 99.85% at Epoch 2. "
        "When evaluated on the untouched, held-out test split of 2,650 samples, the model achieved an overall classification accuracy "
        "of 99.77% (2,644 / 2,650 correct classifications, committing only 6 errors across the entire test set)."
    )
    add_p("Table 5 presents the per-class precision, recall, and F1 scores, while Figure 5 displays the final confusion matrix.")
    
    t5_headers = ["Intent Class", "Precision (%)", "Recall (%)", "F1-Score (%)", "Support (Test Count)", "Test Accuracy (%)"]
    t5_widths = [Inches(1.3), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.1)]
    t5_rows = [
        ["MATHDS", "99.76%", "100.00%", "99.88%", "842", "100.00% (842/842)"],
        ["TXTSRC", "99.69%", "99.69%", "99.69%", "647", "99.69% (645/647)"],
        ["SEMADPT", "99.68%", "99.68%", "99.68%", "617", "99.68% (615/617)"],
        ["CSTRSET", "100.00%", "99.63%", "99.82%", "544", "99.63% (542/544)"],
        ["Macro Average", "99.78%", "99.75%", "99.77%", "2,650", "—"],
        ["Weighted Average", "99.77%", "99.77%", "99.77%", "2,650", "99.77% (Overall)"]
    ]
    t5 = doc.add_table(rows=1, cols=6)
    format_table(t5, t5_widths, t5_headers, t5_rows)

    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig5_confusion_matrix_final.png", 
                  "Figure 5: Confusion matrix of the final DistilBERT classifier evaluated on the N=2,650 test set.")

    add_h2("11.1 Dedicated Boundary Evaluation (Suite 2, N = 200)")
    add_p(
        "To rigorously stress-test the model's boundary discrimination without data leakage, a dedicated holdout suite of 200 adversarial "
        "cases (100 SEMADPT, 100 CSTRSET) containing ambiguous constraint vocabulary was evaluated. On this adversarial set, the classifier "
        "achieved 79.00% accuracy (158/200). All 42 misclassifications were asymmetric: SEMADPT prompts containing rhyming, turn, or sentence "
        "constraints were predicted as CSTRSET, while CSTRSET prompts achieved 100% recall (0 misclassifications into SEMADPT). This established "
        "that while general classification fidelity exceeds 99.7%, highly constrained creative prompts remain vulnerable to schema bias."
    )

    # =========================================================================
    # 13. SPECIALIZED HANDLERS
    # =========================================================================
    add_h1("13. Specialized Handlers")
    add_p(
        "IntentFlow links classifier decisions directly to four specialized execution handlers, each architected around a specific "
        "computational paradigm:"
    )
    add_h2("12.1 MATHDS — Deterministic Math Solver (MathSolverHandler)")
    add_p(
        "The mathematical handler completely replaces probabilistic next-token generation with a deterministic symbolic execution "
        "engine built on Python's SymPy library. The handler extracts mathematical expressions, normalizes implicit algebraic operators, "
        "and executes exact arithmetic, polynomial factorization, root finding, and prime factorization. Because calculation is performed "
        "symbolically on the CPU, arithmetic hallucination is mathematically impossible (0.0% error rate on supported grammars), and execution "
        "completes in an average of 1.04 ms."
    )
    add_h2("12.2 TXTSRC — Grounded Reference Retrieval (SearchRAGHandler)")
    add_p(
        "Rather than relying on the latent parametric memory of an LLM, the factual handler operates over a controlled reference corpus. "
        "When a query arrives, the handler performs lexical and semantic passage lookup across pre-indexed documents, extracts the relevant "
        "supporting context, and formats the output with explicit provenance citations (e.g., citing [REF_03] for Apollo 11 historical details). "
        "This guarantees verifiable provenance and eliminates unsupported confabulation."
    )
    add_h2("12.3 SEMADPT — Cloud LLM Generator (LLMGeneratorHandler)")
    add_p(
        "Queries classified as SEMADPT involve open-ended creative adaptation, stylistic tone rewriting, or narrative synthesis—tasks "
        "where symbolic rules fail and broad parametric capability is essential. These queries are dispatched to Google Gemini 3.1 Flash-Lite "
        "via the official Google GenAI SDK. To balance creativity with coherence, the handler operates at temperature=0.2 with a system "
        "prompt enforcing conciseness. Complete token counts and financial costs are tracked per invocation."
    )
    add_h2("12.4 CSTRSET — Constrained Schema Solver (ConstrainedSolverHandler)")
    add_p(
        "The constraint handler addresses structural formatting tasks (JSON, XML, YAML, delimited arrays). Rather than prompting an LLM "
        "and hoping it does not emit conversational preambles or markdown fences, the handler synthesizes schema structures deterministically. "
        "It validates key-value pairings, enforces strict type constraints, and guarantees syntactic validity before returning, completely "
        "preventing schema drift."
    )

    # =========================================================================
    # 14. ROUTING ARCHITECTURE
    # =========================================================================
    add_h1("14. Routing Architecture")
    add_p(
        "The dispatch logic is coordinated by QueryRouter and ResponseAggregator. The QueryRouter acts as an intelligent switch: it accepts "
        "the user prompt, invokes the classifier (or reads the user-provided intent), maps the label through a canonical dispatch table, "
        "and invokes the designated handler. The ResponseAggregator standardizes execution payloads, capturing execution latencies, token counts, "
        "financial costs, and provenance metadata into a unified JSON structure."
    )

    # =========================================================================
    # 15. EXPERIMENTAL DESIGN
    # =========================================================================
    add_h1("15. Experimental Design")
    add_p(
        "To rigorously quantify the operational differences between monolithic generation and intent routing, the project conducted a "
        "controlled head-to-head empirical experiment in Phase 6.8 across the frozen 80-query benchmark:"
    )
    add_p(
        "• Condition A (Direct Foundation LLM Baseline): Raw benchmark queries were passed directly to Gemini 3.1 Flash-Lite using standard "
        "production settings (temperature=0.0, max_output_tokens=1024, clean academic system instruction). Every query executed remotely.\n"
        "• Condition B (Intent-Routed System): Identical raw benchmark queries were passed to the frozen IntentFlow pipeline. The DistilBERT "
        "classifier predicted intent, the QueryRouter dispatched execution, and local handlers ran on CPU while SEMADPT calls used Gemini (temp=0.2)."
    )
    add_p("Table 6 details the experimental environment and baseline parameters.")
    
    t6_headers = ["Experimental Dimension", "Direct Gemini Baseline", "Intent-Routed System"]
    t6_widths = [Inches(2.2), Inches(2.1), Inches(2.2)]
    t6_rows = [
        ["Model Architecture", "Gemini 3.1 Flash-Lite (Cloud)", "DistilBERT (66M) + Specialized Handlers"],
        ["Inference Location", "100% Cloud API (Google GenAI)", "77.5% Local CPU, 22.5% Cloud API"],
        ["Temperature", "0.0 (Deterministic greedy)", "0.2 for SEMADPT; N/A for local handlers"],
        ["Max Output Tokens", "1,024 tokens", "1,024 tokens (SEMADPT); Handler-bound"],
        ["Benchmark Queries", "80 Queries (20 per intent)", "Identical 80 Queries (Frozen SHA-256)"],
        ["API Pricing Basis", "$0.075 / 1M prompt; $0.30 / 1M output", "$0.075 / 1M prompt; $0.30 / 1M output"]
    ]
    t6 = doc.add_table(rows=1, cols=3)
    format_table(t6, t6_widths, t6_headers, t6_rows)

    # =========================================================================
    # 16. EVALUATION METRICS
    # =========================================================================
    add_h1("16. Evaluation Metrics")
    add_p(
        "Evaluation was conducted using automated, reproducible evaluators across four quantitative dimensions:"
    )
    add_p(
        "1. Response Correctness (%): Domain-specific correctness criteria: numerical precision for MATHDS, ground-truth reference fact "
        "matching for TXTSRC, rubric and constraint compliance for SEMADPT, and JSON/YAML/XML structural syntax parsing for CSTRSET.\n"
        "2. End-to-End Latency (ms): Total turnaround time per query, broken down into mean, median (P50), and 95th percentile (P95).\n"
        "3. Token Consumption & Cost ($): Total prompt and completion tokens billed, translated into USD cost per query and cost per 1,000 queries.\n"
        "4. API Invocation Ratio: Proportion of queries requiring remote API execution versus local zero-cost processing."
    )

    # =========================================================================
    # 17. FINAL EXPERIMENT RESULTS
    # =========================================================================
    add_h1("17. Final Experiment Results")
    add_p(
        "The empirical results demonstrate a striking divergence between response correctness and operational efficiency. Direct Gemini "
        "achieved a perfect 100.00% correctness score across all 80 benchmark queries. The Intent-Routed System achieved 96.25% correctness "
        "(77/80 queries), committing three failures."
    )
    add_p(
        "However, across all resource and latency metrics, the routed architecture demonstrated overwhelming superiority, slashing token "
        "usage by 67.13%, reducing API operating costs by 68.15%, and dropping mean latency by 73.07% (median latency by 92.77%)."
    )
    add_p("Table 7 summarizes the overall comparative performance, Table 8 breaks down latency profiles, and Table 9 details resource consumption.")
    
    t7_headers = ["Performance Metric", "Direct Gemini Baseline", "Intent-Routed System", "Absolute Difference", "Relative Change (%)"]
    t7_widths = [Inches(2.2), Inches(1.3), Inches(1.3), Inches(1.1), Inches(1.1)]
    t7_rows = [
        ["Correct Responses", "80 / 80", "77 / 80", "-3 queries", "-3.75%"],
        ["Benchmark Accuracy (%)", "100.00%", "96.25%", "-3.75 pp", "-3.75%"],
        ["Intent Routing Accuracy (%)", "N/A (Monolithic)", "97.50% (78/80)", "+97.5 pp", "—"],
        ["External Cloud API Calls", "80 calls", "18 calls", "-62 calls", "-77.50%"],
        ["Local Engine Executions", "0 calls", "62 calls", "+62 calls", "+100.00%"],
        ["Total Billed Tokens", "10,170 tokens", "3,343 tokens", "-6,827 tokens", "-67.13%"],
        ["Total Financial Cost (USD)", "$0.002232", "$0.000711", "-$0.001521", "-68.15%"],
        ["Mean Latency (ms)", "3,266.48 ms", "879.63 ms", "-2,386.85 ms", "-73.07%"],
        ["Median (P50) Latency (ms)", "1,921.85 ms", "139.04 ms", "-1,782.81 ms", "-92.77%"],
        ["95th Percentile Latency (ms)", "8,537.59 ms", "3,550.86 ms", "-4,986.73 ms", "-58.41%"]
    ]
    t7 = doc.add_table(rows=1, cols=5)
    format_table(t7, t7_widths, t7_headers, t7_rows)

    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig6_accuracy_comparison.png", 
                  "Figure 6: Controlled benchmark correctness comparison between Direct Gemini (100.0%) and Routed IntentFlow (96.25%).")

    t8_headers = ["Latency Percentile", "Direct Gemini (ms)", "Intent-Routed (ms)", "Absolute Delta (ms)", "Speedup Factor"]
    t8_widths = [Inches(1.8), Inches(1.3), Inches(1.3), Inches(1.2), Inches(0.9)]
    t8_rows = [
        ["Mean Latency", "3,266.48 ms", "879.63 ms", "-2,386.85 ms", "3.71x Faster"],
        ["Median (P50) Latency", "1,921.85 ms", "139.04 ms", "-1,782.81 ms", "13.82x Faster"],
        ["95th Percentile (P95)", "8,537.59 ms", "3,550.86 ms", "-4,986.73 ms", "2.40x Faster"],
        ["Total Benchmark Run Time", "261.32 sec", "70.37 sec", "-190.95 sec", "3.71x Overall"]
    ]
    t8 = doc.add_table(rows=1, cols=5)
    format_table(t8, t8_widths, t8_headers, t8_rows)

    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig7_latency_comparison.png", 
                  "Figure 7: Latency profiles across Mean, Median (P50), and Tail (P95) percentiles.")

    t9_headers = ["Resource Dimension", "Direct Gemini Baseline", "Intent-Routed System", "Absolute Savings", "Reduction (%)"]
    t9_widths = [Inches(2.2), Inches(1.3), Inches(1.3), Inches(1.1), Inches(1.1)]
    t9_rows = [
        ["Prompt / Input Tokens", "1,698 tokens", "1,043 tokens", "-655 tokens", "-38.58%"],
        ["Completion / Output Tokens", "8,472 tokens", "2,300 tokens", "-6,172 tokens", "-72.85%"],
        ["Total Billed Tokens", "10,170 tokens", "3,343 tokens", "-6,827 tokens", "-67.13%"],
        ["Avg Tokens per Query", "127.1 tokens", "41.8 tokens", "-85.3 tokens", "-67.13%"],
        ["Total Cost (80 Queries)", "$0.002232", "$0.000711", "-$0.001521", "-68.15%"],
        ["Cost per 1,000 Queries", "$0.0279", "$0.0089", "-$0.0190", "-68.15%"]
    ]
    t9 = doc.add_table(rows=1, cols=5)
    format_table(t9, t9_widths, t9_headers, t9_rows)

    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig8_efficiency_comparison.png", 
                  "Figure 8: Comparative resource consumption: Total Tokens (-67.1%), Cost (-68.2%), and API Invocations (-77.5%).")

    # =========================================================================
    # 18. PER-INTENT RESULTS
    # =========================================================================
    add_h1("18. Per-Intent Analysis")
    add_p(
        "Disaggregating performance across the four intent quadrants reveals distinct behavioral dynamics between deterministic and generative routes. "
        "Figure 9 depicts per-intent accuracy, demonstrating that deterministic quadrants achieved flawless execution."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig9_per_intent_accuracy.png", 
                  "Figure 9: Per-intent response correctness comparison across the four benchmark classes (N=20 queries each).")
    add_p(
        "• MATHDS (20/20 Direct vs. 20/20 Routed, 100.0%): In the mathematical domain, the routed system achieved 100% accuracy while reducing "
        "mean latency from 4,028.20 ms to 230.43 ms (a 94.28% speedup) and consuming zero tokens. Local SymPy execution eliminated all arithmetic drift.\n"
        "• TXTSRC (20/20 Direct vs. 19/20 Routed, 95.0%): The routed system routed all 20 queries correctly to SearchRAG. 19 queries produced accurate, "
        "citation-backed answers in 149.25 ms (vs. 4,153.08 ms for direct Gemini). The single failure (TXT_19) was an evaluator naming artifact.\n"
        "• SEMADPT (20/20 Direct vs. 18/20 Routed, 90.0%): Two queries (SEM_08 and SEM_17) were misrouted to CSTRSET, leading to empty/fallback solver "
        "responses. Correctly routed queries executed via Gemini in 3,005.58 ms with full semantic fluency.\n"
        "• CSTRSET (20/20 Direct vs. 20/20 Routed, 100.0%): All 20 structural schema prompts were routed correctly to ConstrainedSolver, achieving "
        "perfect syntax compliance in an average of 133.26 ms (vs. 1,199.50 ms for Gemini) with zero schema drift."
    )

    # =========================================================================
    # 19. FAILURE ANALYSIS
    # =========================================================================
    add_h1("19. Failure Analysis")
    add_p(
        "Scientific transparency requires an exhaustive audit of the three queries where the routed system failed to match the direct baseline. "
        "Figure 10 presents the failure attribution flowchart, categorizing the errors into evaluator artifacts and classifier routing errors."
    )
    add_fig_image(PROJECT_ROOT / "artifacts/figures/fig10_failure_flow.png", 
                  "Figure 10: Root-cause taxonomy of the three routed system failures.")
    
    add_p("Table 10 provides the exact query strings, predicted intents, handler outputs, and root-cause attributions.")
    
    t10_headers = ["Query ID", "Expected Intent", "Predicted Intent", "Assigned Handler", "Root Cause Attribution", "Scientific Finding"]
    t10_widths = [Inches(0.9), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.3), Inches(1.2)]
    t10_rows = [
        ["TXT_19", "TXTSRC", "TXTSRC (99.96%)", "rag_search", "Evaluator Substring Artifact", "Response cited 'Dorothy Crowfoot Hodgkin' from corpus; evaluator strictly checked 'Dorothy Hodgkin'."],
        ["SEM_08", "SEMADPT", "CSTRSET (86.51%)", "constrained_solver", "Classifier Routing Error", "Rhyming poem with AABB constraint triggered CSTRSET prior; solver produced fallback notice."],
        ["SEM_17", "SEMADPT", "CSTRSET (99.90%)", "constrained_solver", "Classifier Routing Error", "Haiku with 5-7-5 syllable constraint triggered CSTRSET prior; solver produced fallback notice."]
    ]
    t10 = doc.add_table(rows=1, cols=6)
    format_table(t10, t10_widths, t10_headers, t10_rows)

    add_p(
        "Crucially, TXT_19 was NOT a model hallucination. The SearchRAG handler retrieved the correct historical passage (REF_19) and correctly "
        "stated that the first British woman to win a Nobel Prize was Dorothy Crowfoot Hodgkin (her formal biographical name). However, the automated "
        "benchmark regex strictly checked for the exact substring 'Dorothy Hodgkin'. In contrast, SEM_08 and SEM_17 represent genuine classifier "
        "boundary limitations: explicit structural rules (rhyme scheme, syllable counts) overpowered the creative generation intent."
    )

    # =========================================================================
    # 20. MATCHED-PAIR ANALYSIS
    # =========================================================================
    add_h1("20. Matched-Pair Analysis")
    add_p("A query-by-query matched-pair comparison across the 80 benchmark queries reveals the following distribution:")
    add_p(
        "• Both Systems Succeeded: 77 queries (96.25%)\n"
        "• Only Direct Gemini Succeeded: 3 queries (3.75% — TXT_19, SEM_08, SEM_17)\n"
        "• Only Routed System Succeeded: 0 queries (0.00%)\n"
        "• Both Systems Failed: 0 queries (0.00%)"
    )
    add_p(
        "Descriptively, this establishes that the direct foundation model achieved superior raw correctness on this specific evaluation set. "
        "The routed system did not surpass Gemini in domain correctness on any benchmark subset, but matched it completely in MATHDS and CSTRSET."
    )

    # =========================================================================
    # 21. EFFICIENCY ANALYSIS
    # =========================================================================
    add_h1("21. Efficiency Analysis")
    add_p(
        "The primary justification for intent routing lies in computational and financial efficiency. Monolithic LLM deployment forces "
        "every query through hundreds of layers of self-attention and network serialization. By contrast, IntentFlow offloaded 77.50% (62/80) "
        "of all queries to local CPU handlers. This yields dramatic latency divergence:"
    )
    add_p(
        "While direct Gemini required an average of 3,266.48 ms per query, the routed system completed queries in an average of 879.63 ms. "
        "Even more pronounced is the median (P50) latency: Direct Gemini's median was 1,921.85 ms, whereas the routed system achieved a median "
        "of 139.04 ms—a 13.82x speedup (92.77% reduction). For local mathematical and constraint queries, latency drops below 2 milliseconds "
        "once classification completes."
    )

    # =========================================================================
    # 22. COST–ACCURACY–LATENCY TRADEOFF
    # =========================================================================
    add_h1("22. Cost–Accuracy–Latency Tradeoff")
    add_p(
        "The central conceptual synthesis of this study is the formalization of the Cost–Accuracy–Latency Tradeoff in AI system design. "
        "Commercial AI literature frequently claims that small, specialized models or multi-agent routing systems can universally outperform "
        "foundation models in both accuracy and cost. Our empirical findings refute this oversimplification, demonstrating instead a Pareto boundary:"
    )
    add_callout(
        doc,
        "The Tradeoff Thesis: Introducing an intent routing layer provides enormous efficiency benefits—reducing API expenditure by 68.15%, "
        "token consumption by 67.13%, and median latency by 92.77%—at the cost of a modest 3.75 percentage point decrease in benchmark accuracy "
        "(96.25% vs. 100.00%). The decision to deploy intent routing is therefore an architectural tradeoff between operating cost and boundary accuracy.",
        title="The Cost-Accuracy Tradeoff Thesis",
        border_color="D97706", bg_color="FFFBEB"
    )

    # =========================================================================
    # 23. DEVELOPMENT AND BOUNDARY EVOLUTION
    # =========================================================================
    add_h1("23. Development and Boundary Evolution")
    add_p(
        "The final IntentFlow architecture evolved across multiple research iterations documented in the project record:"
    )
    add_p(
        "• Phase 5: Initial DistilBERT fine-tuning revealed substantial confusion between SEMADPT and CSTRSET on complex prompts.\n"
        "• Phase 5.1: Introduction of distinct taxonomy definitions and initial rule restructuring.\n"
        "• Phase 6.3 & 6.4: Major dataset expansion incorporating GSM8K, CLINC150, RECAST, XSum, and TriviaQA, scaling data to 25,692 samples (V2).\n"
        "• Phase 6.5 & 6.6: End-to-end evaluation revealed 5 SEMADPT routing failures (SEM_03, 08, 12, 16, 17) caused by IFEval constraint bias.\n"
        "• Phase 6.7: Targeted boundary intervention adding 800 paired boundary examples (Dataset V3, N=26,492). Successfully fixed 3 of the 5 "
        "failures (SEM_03, 12, 16), elevating benchmark accuracy from 92.50% to 96.25%.\n"
        "• Phase 6.8: Final model freeze, comprehensive 80-query benchmark execution, and artifact consolidation."
    )

    # =========================================================================
    # 24. LIMITATIONS
    # =========================================================================
    add_h1("24. Limitations")
    add_p("A rigorous scientific inquiry must acknowledge its methodological and experimental limitations:")
    add_p(
        "1. Benchmark Sample Size: The evaluation benchmark comprises 80 carefully curated queries. While highly informative for controlled comparison, "
        "statistical confidence intervals are wider than those derived from thousand-query benchmarks.\n"
        "2. Evaluator Rigidity: As demonstrated by query TXT_19, automated substring matching can penalize responses that provide more complete "
        "historical names ('Dorothy Crowfoot Hodgkin') than the expected gold key ('Dorothy Hodgkin').\n"
        "3. Semantic-Constraint Boundary Vulnerability: Despite boundary augmentation in Phase 6.7, prompts combining creative verse with explicit "
        "meter/syllable constraints (SEM_08, SEM_17) remain prone to CSTRSET misclassification.\n"
        "4. Generative Dependence: SEMADPT relies on an external foundation LLM; the architecture reduces, but does not completely eliminate, "
        "cloud API dependencies.\n"
        "5. CPU Inference Profile: Benchmarks were conducted in a local CPU environment. While demonstrating edge feasibility, batched GPU inference "
        "would alter absolute throughput metrics."
    )

    # =========================================================================
    # 25. THREATS TO VALIDITY
    # =========================================================================
    add_h1("25. Threats to Validity")
    add_p(
        "• Internal Validity: Mitigated by cryptographically freezing benchmark queries and reference corpora, running identical prompts across both "
        "systems, and isolating training splits from evaluation sets with 0% data leakage.\n"
        "• External Validity: The four-intent taxonomy captures major query archetypes but may require expansion for multimodal, conversational memory, "
        "or multi-turn agentic workflows.\n"
        "• Construct Validity: Response correctness was evaluated using deterministic unit tests, reference citations, and JSON schema validators, "
        "minimizing subjective human grading bias."
    )

    # =========================================================================
    # 26. PRACTICAL IMPLICATIONS
    # =========================================================================
    add_h1("26. Practical Implications")
    add_p(
        "For system architects and enterprise engineers, these findings offer vital practical guidance: intent-aware routing provides an immediate, "
        "low-overhead mechanism to reduce LLM operational expenditure by over 68% and drop median latency below 150 ms. For organizations deploying "
        "customer-facing assistants, routing mathematical and structured queries to deterministic local handlers eliminates compliance and calculation "
        "hallucinations while preserving commercial API budgets for tasks that genuinely require human-like semantic fluency."
    )

    # =========================================================================
    # 27. FUTURE WORK
    # =========================================================================
    add_h1("27. Future Work")
    add_p("Future research directions arising from this study include:")
    add_p(
        "1. Confidence-Aware Generative Fallback: Implementing an entropy threshold where boundary queries with confidence <90% automatically fall back "
        "to the cloud LLM, eliminating failure modes like SEM_08 and SEM_17.\n"
        "2. Multi-Label / Hierarchical Intent Classification: Decomposing prompts into primary communicative intent (creative generation) and secondary "
        "surface constraints (syllable counts).\n"
        "3. Semantic Fuzzy Evaluation: Upgrading benchmark evaluators with embedding-based semantic equivalence checking to resolve naming artifacts.\n"
        "4. Dynamic Schema Expansion: Enhancing the local ConstrainedSolver to dynamically parse arbitrary user schemas without static templates."
    )

    # =========================================================================
    # 28. CONCLUSION
    # =========================================================================
    add_h1("28. Conclusion")
    add_p(
        "This project designed, implemented, and empirically benchmarked IntentFlow, an intent-aware routing framework for Large Language Model "
        "systems. Driven by a fine-tuned DistilBERT classifier (66M parameters) trained on 26,492 multi-source examples, the system accurately "
        "segregates queries into four computational intents: mathematical reasoning, factual retrieval, semantic adaptation, and constraint enforcement."
    )
    add_p(
        "In a controlled head-to-head benchmark against Google Gemini 3.1 Flash-Lite, the direct baseline achieved 100.00% correctness, while the "
        "routed architecture achieved 96.25% correctness. Crucially, the routed system slashed total token usage by 67.13%, reduced API costs by 68.15%, "
        "offloaded 77.50% of queries to zero-cost local engines, and reduced median response latency by 92.77% (from 1,921.85 ms to 139.04 ms). "
        "These findings conclusively establish that intent routing is a powerful, economically transformative architectural paradigm, characterized by "
        "a distinct and measurable cost-accuracy-latency tradeoff."
    )

    # =========================================================================
    # REFERENCES
    # =========================================================================
    doc.add_page_break()
    add_comprehensive_references_section(doc, add_h1, add_p)

    # Save DOCX
    docx_path = PROJECT_ROOT / "artifacts/reports/MASTER_TECHNICAL_RESEARCH_REPORT.docx"
    doc.save(str(docx_path))
    print(f"[SUCCESS] Saved DOCX report to: {docx_path}")

    # Export to PDF via MS Word COM if available
    pdf_path = PROJECT_ROOT / "artifacts/reports/MASTER_TECHNICAL_RESEARCH_REPORT.pdf"
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        doc_com = word.Documents.Open(str(docx_path.resolve()))
        # FileFormat=17 represents wdFormatPDF
        doc_com.SaveAs(str(pdf_path.resolve()), FileFormat=17)
        doc_com.Close()
        word.Quit()
        print(f"[SUCCESS] Exported PDF report to: {pdf_path}")
    except Exception as e:
        print(f"[WARNING] Word COM PDF export failed: {e}")

if __name__ == '__main__':
    create_report()
