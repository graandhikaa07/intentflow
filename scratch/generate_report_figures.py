import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

os.makedirs('artifacts/figures', exist_ok=True)

# Set clean academic style
plt.rcParams.update({
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'figure.dpi': 300
})

PRIMARY_COLOR = '#1f77b4'   # Academic blue
SECONDARY_COLOR = '#2ca02c' # Academic green
ACCENT_COLOR = '#d62728'    # Academic red
DIRECT_COLOR = '#4a5568'    # Slate gray
ROUTED_COLOR = '#0284c7'    # Vibrant blue
BORDER_COLOR = '#2d3748'

# ==============================================================================
# FIGURE 1: Overall System Architecture
# ==============================================================================
def generate_figure_1():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axis('off')
    
    # Outer frame
    rect_main = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.02",
                                        facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(rect_main)
    
    # Header Title inside diagram
    ax.text(0.5, 0.91, "IntentFlow System Architecture: Dual-Mode Intent-Aware Routing Pipeline", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')
    
    # 1. User Input Box
    box_user = patches.FancyBboxPatch((0.05, 0.45), 0.16, 0.32, boxstyle="round,pad=0.01",
                                      facecolor="#ffffff", edgecolor="#0284c7", linewidth=2)
    ax.add_patch(box_user)
    ax.text(0.13, 0.70, "User Query", ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')
    ax.text(0.13, 0.58, "Query Input\nComposer", ha='center', va='center', fontsize=9, color='#475569')
    ax.text(0.13, 0.48, "[ Intent Selector ]\n(Auto / Manual)", ha='center', va='center', fontsize=8, color='#0284c7', style='italic')

    # 2. Mode Splitter
    ax.annotate("", xy=(0.25, 0.65), xytext=(0.21, 0.65),
                arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2))
    ax.text(0.23, 0.67, "Auto", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#0284c7")

    ax.annotate("", xy=(0.46, 0.38), xytext=(0.21, 0.50),
                arrowprops=dict(arrowstyle="->", color="#d97706", lw=2, linestyle="--"))
    ax.text(0.31, 0.45, "User-Provided Intent", ha='center', va='bottom', fontsize=8, fontweight='bold', color="#d97706")

    # 3. DistilBERT Classifier Box (Auto Mode)
    box_clf = patches.FancyBboxPatch((0.25, 0.54), 0.14, 0.23, boxstyle="round,pad=0.01",
                                     facecolor="#eff6ff", edgecolor="#3b82f6", linewidth=1.5)
    ax.add_patch(box_clf)
    ax.text(0.32, 0.69, "DistilBERT\nClassifier", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1e3a8a')
    ax.text(0.32, 0.60, "66M Parameters\nFine-Tuned SeqClf\n4-Class Softmax", ha='center', va='center', fontsize=7.5, color='#1e40af')

    # Arrow from Classifier to Router
    ax.annotate("", xy=(0.46, 0.65), xytext=(0.395, 0.65),
                arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2))
    ax.text(0.428, 0.68, "Pred", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color="#3b82f6")

    # 4. Query Router Box
    box_router = patches.FancyBboxPatch((0.465, 0.32), 0.125, 0.46, boxstyle="round,pad=0.01",
                                       facecolor="#f1f5f9", edgecolor="#475569", linewidth=2)
    ax.add_patch(box_router)
    ax.text(0.527, 0.72, "QueryRouter", ha='center', va='center', fontsize=11, fontweight='bold', color='#0f172a')
    ax.text(0.527, 0.55, "Dispatch Engine\n\n- Canonical Map\n- Telemetry Log\n- Venue Select", ha='center', va='center', fontsize=8, color='#334155')

    # 5. Four Specialized Handlers
    handlers = [
        ("MATHDS", "Math Solver (SymPy)", "Local Deterministic Engine", "#dcfce7", "#15803d", 0.74),
        ("TXTSRC", "Grounded RAG Store", "Reference Passage Search", "#e0e7ff", "#4338ca", 0.56),
        ("SEMADPT", "Cloud LLM Generator", "Gemini 3.1 Flash-Lite (API)", "#fef3c7", "#b45309", 0.38),
        ("CSTRSET", "Constrained Solver", "Schema / Format Synthesizer", "#f3e8ff", "#7e22ce", 0.20)
    ]
    
    for code, name, desc, bg, border, y_pos in handlers:
        # Arrow from router to handler
        ax.annotate("", xy=(0.63, y_pos + 0.05), xytext=(0.59, 0.55),
                    arrowprops=dict(arrowstyle="->", color=border, lw=1.5))
        # Handler Box
        h_box = patches.FancyBboxPatch((0.63, y_pos), 0.20, 0.12, boxstyle="round,pad=0.008",
                                       facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(h_box)
        ax.text(0.73, y_pos + 0.08, f"[{code}] {name}", ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a')
        ax.text(0.73, y_pos + 0.035, desc, ha='center', va='center', fontsize=7.5, color='#334155')

        # Arrow from handler to aggregator
        ax.annotate("", xy=(0.86, 0.55), xytext=(0.83, y_pos + 0.05),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

    # 6. Response Aggregator Box
    box_agg = patches.FancyBboxPatch((0.86, 0.35), 0.10, 0.40, boxstyle="round,pad=0.01",
                                     facecolor="#ffffff", edgecolor="#0f172a", linewidth=2)
    ax.add_patch(box_agg)
    ax.text(0.91, 0.68, "Response\nAggregator", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax.text(0.91, 0.50, "Output Formatter\nMetadata Merge\nToken/Latency\nAccounting", ha='center', va='center', fontsize=7.5, color='#475569')

    # Output Arrow
    ax.annotate("", xy=(0.98, 0.55), xytext=(0.96, 0.55),
                arrowprops=dict(arrowstyle="->", color="#0f172a", lw=2))
    
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig1_system_architecture.png', dpi=300)
    plt.close()
    print("Saved fig1_system_architecture.png")

# ==============================================================================
# FIGURE 2: End-to-End Routing Flow Diagram
# ==============================================================================
def generate_figure_2():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axis('off')
    
    rect_main = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.02",
                                        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(rect_main)

    ax.text(0.5, 0.93, "End-to-End Decision & Execution Flowchart", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')

    # Steps
    steps = [
        ("1. Query Ingestion", "User submits query with optional intent selector (Default: Auto-detect)", 0.83, "#f8fafc", "#64748b"),
        ("2. Routing Decision Gate", "Check intent mode: Is explicit intent supplied by user?", 0.70, "#fef3c7", "#d97706"),
    ]
    for title, desc, y, bg, border in steps:
        b = patches.FancyBboxPatch((0.15, y - 0.04), 0.70, 0.08, boxstyle="round,pad=0.01", facecolor=bg, edgecolor=border, lw=1.5)
        ax.add_patch(b)
        ax.text(0.50, y + 0.015, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#0f172a')
        ax.text(0.50, y - 0.018, desc, ha='center', va='center', fontsize=8.5, color='#334155')

    ax.annotate("", xy=(0.50, 0.74), xytext=(0.50, 0.79), arrowprops=dict(arrowstyle="->", color="#0f172a", lw=2))

    # Branching
    # Left: Auto-detect branch
    ax.annotate("", xy=(0.30, 0.55), xytext=(0.35, 0.66), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2))
    ax.text(0.28, 0.61, "Auto-detect\n(Default)", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#2563eb")

    b_auto = patches.FancyBboxPatch((0.10, 0.44), 0.36, 0.11, boxstyle="round,pad=0.01", facecolor="#eff6ff", edgecolor="#2563eb", lw=1.5)
    ax.add_patch(b_auto)
    ax.text(0.28, 0.51, "DistilBERT Classifier Runs", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1e3a8a')
    ax.text(0.28, 0.465, "Output: Intent Label + Softmax Confidence\nMeasured Clf Latency (~41-70 ms)", ha='center', va='center', fontsize=8, color='#1e40af')

    # Right: User-provided branch
    ax.annotate("", xy=(0.70, 0.55), xytext=(0.65, 0.66), arrowprops=dict(arrowstyle="->", color="#d97706", lw=2))
    ax.text(0.72, 0.61, "Explicit Intent\nSelected", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#d97706")

    b_user = patches.FancyBboxPatch((0.54, 0.44), 0.36, 0.11, boxstyle="round,pad=0.01", facecolor="#fffbeb", edgecolor="#d97706", lw=1.5)
    ax.add_patch(b_user)
    ax.text(0.72, 0.51, "Classifier Bypassed", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400e')
    ax.text(0.72, 0.465, "Decision Source: 'user_provided'\nLatency: 0.0 ms | Confidence: None (omitted)", ha='center', va='center', fontsize=8, color='#b45309')

    # Merge into Dispatch
    ax.annotate("", xy=(0.50, 0.34), xytext=(0.28, 0.44), arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5))
    ax.annotate("", xy=(0.50, 0.34), xytext=(0.72, 0.44), arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5))

    b_disp = patches.FancyBboxPatch((0.15, 0.24), 0.70, 0.09, boxstyle="round,pad=0.01", facecolor="#f8fafc", edgecolor="#475569", lw=1.5)
    ax.add_patch(b_disp)
    ax.text(0.50, 0.295, "3. QueryRouter Dispatch & Execution", ha='center', va='center', fontsize=10, fontweight='bold', color='#0f172a')
    ax.text(0.50, 0.26, "Routes to designated handler (MathSolver / SearchRAG / LLMGen / ConstrainedSolver)", ha='center', va='center', fontsize=8.5, color='#334155')

    ax.annotate("", xy=(0.50, 0.16), xytext=(0.50, 0.24), arrowprops=dict(arrowstyle="->", color="#0f172a", lw=2))

    b_out = patches.FancyBboxPatch((0.15, 0.07), 0.70, 0.09, boxstyle="round,pad=0.01", facecolor="#f0fdf4", edgecolor="#16a34a", lw=1.5)
    ax.add_patch(b_out)
    ax.text(0.50, 0.125, "4. Response Aggregation & Telemetry Delivery", ha='center', va='center', fontsize=10, fontweight='bold', color='#14532d')
    ax.text(0.50, 0.09, "User receives verified response + transparent provenance and execution breakdown", ha='center', va='center', fontsize=8.5, color='#166534')

    plt.tight_layout()
    plt.savefig('artifacts/figures/fig2_routing_flow.png', dpi=300)
    plt.close()
    print("Saved fig2_routing_flow.png")

# ==============================================================================
# FIGURE 3: Four-Intent Taxonomy & Handler Mapping
# ==============================================================================
def generate_figure_3():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')
    
    rect_main = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.02",
                                        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(rect_main)

    ax.text(0.5, 0.92, "Four-Intent Taxonomy & Handler Mapping Matrix", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')

    tax_cards = [
        ("MATHDS", "Mathematical Reasoning", "Deterministic Local Math Solver", "SymPy execution engine",
         "Arithmetic, algebraic equations, prime factorization, statistical calculation",
         "100.0% deterministic accuracy; zero arithmetic drift or hallucination",
         "#dcfce7", "#16a34a", 0.06, 0.48),
        ("TXTSRC", "Factual Information", "Grounded Corpus Search-RAG", "Controlled reference store lookup",
         "Historical events, scientific constants, capital cities, biographical dates",
         "Strict provenance attribution via [REF_XX] citations from reference corpus",
         "#e0e7ff", "#4338ca", 0.52, 0.48),
        ("SEMADPT", "Semantic / Creative", "Cloud Foundation LLM Generator", "Gemini 3.1 Flash-Lite (temp=0.2)",
         "Creative narratives, open poems, tone rewriting, stylistic dialogue adaptation",
         "High semantic fluency and open-ended generative capability via external LLM",
         "#fef3c7", "#d97706", 0.06, 0.08),
        ("CSTRSET", "Constraint Enforcement", "Deterministic Constrained Solver", "Local schema/format synthesizer",
         "JSON schemas, XML trees, YAML structures, delimited item count arrays",
         "Guaranteed syntax compliance; eliminates structural drift and markdown leaking",
         "#f3e8ff", "#9333ea", 0.52, 0.08)
    ]

    for code, title, handler, tech, queries, guarantee, bg, border, x, y in tax_cards:
        card = patches.FancyBboxPatch((x, y), 0.42, 0.36, boxstyle="round,pad=0.015",
                                      facecolor=bg, edgecolor=border, lw=1.5)
        ax.add_patch(card)
        ax.text(x + 0.02, y + 0.32, f"[{code}] {title}", fontsize=10, fontweight='bold', color='#0f172a')
        ax.text(x + 0.02, y + 0.26, f"Handler: {handler}", fontsize=8.5, fontweight='bold', color=border)
        ax.text(x + 0.02, y + 0.21, f"Mechanism: {tech}", fontsize=8, color='#334155')
        ax.text(x + 0.02, y + 0.14, f"Typical Queries: {queries}", fontsize=7.5, color='#475569', wrap=True)
        ax.text(x + 0.02, y + 0.05, f"Execution Guarantee: {guarantee}", fontsize=7.5, color='#0f172a', style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig('artifacts/figures/fig3_intent_taxonomy.png', dpi=300)
    plt.close()
    print("Saved fig3_intent_taxonomy.png")

# ==============================================================================
# FIGURE 4: Final Dataset Class Distribution
# ==============================================================================
def generate_figure_4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    classes = ['MATHDS', 'TXTSRC', 'SEMADPT', 'CSTRSET']
    counts = [8413, 6471, 6168, 5440]
    percentages = [31.76, 24.43, 23.28, 20.53]
    colors = ['#16a34a', '#2563eb', '#d97706', '#9333ea']

    # Bar chart
    bars = ax1.bar(classes, counts, color=colors, edgecolor='#1e293b', width=0.55, linewidth=1.2)
    ax1.set_title("Dataset V3 Class Counts (N = 26,492)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Number of Query Examples")
    ax1.set_ylim(0, 9800)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar, pct, cnt in zip(bars, percentages, counts):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 180, f"{cnt:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Donut chart for Partition Splits
    split_labels = ['Train (80%)', 'Validation (10%)', 'Held-out Test (10%)']
    split_counts = [21193, 2649, 2650]
    split_colors = ['#0284c7', '#38bdf8', '#bae6fd']
    
    wedges, texts, autotexts = ax2.pie(split_counts, labels=split_labels, autopct='%1.1f%%',
                                       startangle=140, colors=split_colors,
                                       wedgeprops=dict(width=0.4, edgecolor='#ffffff', linewidth=2))
    for t in texts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color('#0f172a')
        
    ax2.set_title("Dataset Partition Splits", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig4_dataset_distribution.png', dpi=300)
    plt.close()
    print("Saved fig4_dataset_distribution.png")

# ==============================================================================
# FIGURE 5: Final DistilBERT Confusion Matrix
# ==============================================================================
def generate_figure_5():
    # Phase 6.7 final test set confusion matrix (N=2,650)
    labels = ['MATHDS', 'TXTSRC', 'SEMADPT', 'CSTRSET']
    cm = np.array([
        [842,   0,   0,   0],
        [  2, 645,   0,   0],
        [  0,   2, 615,   0],
        [  0,   0,   2, 542]
    ])
    
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=1, linecolor='#cbd5e1', annot_kws={"size": 11, "fontweight": "bold"})
    
    ax.set_title("DistilBERT Final Confusion Matrix\n(Held-out Test Set, N = 2,650 | Accuracy = 99.77%)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Intent Class", fontweight='bold')
    ax.set_ylabel("True Intent Class", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig5_confusion_matrix_final.png', dpi=300)
    plt.close()
    print("Saved fig5_confusion_matrix_final.png")

# ==============================================================================
# FIGURE 6: Benchmark Direct vs Routed Accuracy
# ==============================================================================
def generate_figure_6():
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    
    models = ['Direct Gemini 3.1 Flash-Lite', 'Intent-Routed System']
    acc = [100.0, 96.25]
    colors = [DIRECT_COLOR, ROUTED_COLOR]
    
    bars = ax.bar(models, acc, color=colors, width=0.45, edgecolor='#1e293b', linewidth=1.2)
    ax.set_ylabel("Benchmark Response Correctness (%)", fontweight='bold')
    ax.set_title("Controlled Benchmark Correctness\n(80-Query Evaluation Benchmark)", fontsize=11, fontweight='bold')
    ax.set_ylim(80, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar, val in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.8, f"{val:.2f}%\n({int(val*0.8)}/80)", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
        
    # Annotate the 3 errors
    ax.annotate("3 Errors:\n- 1 Evaluator substring artifact (TXT_19)\n- 2 Creative boundary misroutes (SEM_08, 17)",
                xy=(1, 96.25), xytext=(0.4, 85),
                arrowprops=dict(facecolor='#d97706', shrink=0.08, width=1.2, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffbeb", ec="#d97706", lw=1),
                fontsize=8)
    
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig6_accuracy_comparison.png', dpi=300)
    plt.close()
    print("Saved fig6_accuracy_comparison.png")

# ==============================================================================
# FIGURE 7: Latency Comparison (Mean, P50, P95)
# ==============================================================================
def generate_figure_7():
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    metrics = ['Mean Latency', 'Median (P50)', '95th Percentile (P95)']
    direct_lat = [3266.48, 1921.85, 8537.59]
    routed_lat = [879.63, 139.04, 3550.86]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    b1 = ax.bar(x - width/2, direct_lat, width, label='Direct Gemini', color=DIRECT_COLOR, edgecolor='#1e293b')
    b2 = ax.bar(x + width/2, routed_lat, width, label='Routed System', color=ROUTED_COLOR, edgecolor='#1e293b')
    
    ax.set_ylabel("Latency (milliseconds)", fontweight='bold')
    ax.set_title("End-to-End Latency Profile: Direct vs. Routed System", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Reductions
    reductions = ["-73.1%", "-92.8%", "-58.4%"]
    for i in range(len(metrics)):
        diff = direct_lat[i] - routed_lat[i]
        ax.text(x[i] + width/2, routed_lat[i] + 150, f"{routed_lat[i]:.1f} ms\n({reductions[i]})", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0369a1')
        ax.text(x[i] - width/2, direct_lat[i] + 150, f"{direct_lat[i]:.1f} ms", ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, 9600)
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig7_latency_comparison.png', dpi=300)
    plt.close()
    print("Saved fig7_latency_comparison.png")

# ==============================================================================
# FIGURE 8: Token, Cost, and API Call Comparison
# ==============================================================================
def generate_figure_8():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4.2))
    
    systems = ['Direct', 'Routed']
    
    # 1. Total Tokens
    tokens = [10170, 3343]
    b1 = ax1.bar(systems, tokens, color=[DIRECT_COLOR, ROUTED_COLOR], width=0.5, edgecolor='#1e293b')
    ax1.set_title("Total Token Usage\n(-67.1% Reduction)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Tokens")
    ax1.set_ylim(0, 11800)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for b in b1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 200, f"{int(b.get_height()):,}", ha='center', fontweight='bold', fontsize=8.5)

    # 2. Total Cost
    cost = [0.002232, 0.000711]
    b2 = ax2.bar(systems, cost, color=[DIRECT_COLOR, ROUTED_COLOR], width=0.5, edgecolor='#1e293b')
    ax2.set_title("Total API Cost (USD)\n(-68.2% Reduction)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("Cost ($)")
    ax2.set_ylim(0, 0.0026)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for b in b2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.00005, f"${b.get_height():.6f}", ha='center', fontweight='bold', fontsize=8.5)

    # 3. Gemini API Calls
    calls = [80, 18]
    b3 = ax3.bar(systems, calls, color=[DIRECT_COLOR, ROUTED_COLOR], width=0.5, edgecolor='#1e293b')
    ax3.set_title("External API Calls\n(-77.5% Offloaded)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Number of API Calls")
    ax3.set_ylim(0, 95)
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    for b in b3:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{int(b.get_height())}/80", ha='center', fontweight='bold', fontsize=8.5)
        
    plt.tight_layout()
    plt.savefig('artifacts/figures/fig8_efficiency_comparison.png', dpi=300)
    plt.close()
    print("Saved fig8_efficiency_comparison.png")

# ==============================================================================
# FIGURE 9: Per-Intent Accuracy Comparison
# ==============================================================================
def generate_figure_9():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    
    intents = ['MATHDS\n(Math Solver)', 'TXTSRC\n(Search RAG)', 'SEMADPT\n(LLM Gen)', 'CSTRSET\n(Constrained)']
    direct_acc = [100.0, 100.0, 100.0, 100.0]
    routed_acc = [100.0,  95.0,  90.0, 100.0]
    
    x = np.arange(len(intents))
    width = 0.35
    
    b1 = ax.bar(x - width/2, direct_acc, width, label='Direct Gemini', color=DIRECT_COLOR, edgecolor='#1e293b')
    b2 = ax.bar(x + width/2, routed_acc, width, label='Routed System', color=ROUTED_COLOR, edgecolor='#1e293b')
    
    ax.set_ylabel("Response Accuracy (%)", fontweight='bold')
    ax.set_title("Per-Intent Response Correctness Comparison (N=20 per class)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(intents, fontweight='bold')
    ax.set_ylim(75, 108)
    ax.legend(loc='lower left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    for i in range(len(intents)):
        ax.text(x[i] - width/2, direct_acc[i] + 0.8, f"{direct_acc[i]:.0f}%", ha='center', fontsize=8.5, fontweight='bold')
        ax.text(x[i] + width/2, routed_acc[i] + 0.8, f"{routed_acc[i]:.0f}%", ha='center', fontsize=8.5, fontweight='bold',
                color='#15803d' if routed_acc[i] == 100 else '#b45309')

    plt.tight_layout()
    plt.savefig('artifacts/figures/fig9_per_intent_accuracy.png', dpi=300)
    plt.close()
    print("Saved fig9_per_intent_accuracy.png")

# ==============================================================================
# FIGURE 10: Failure Analysis Flowchart
# ==============================================================================
def generate_figure_10():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axis('off')
    
    rect_main = patches.FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.02",
                                        facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5)
    ax.add_patch(rect_main)

    ax.text(0.5, 0.92, "Failure Attribution & Error Taxonomy (3 Discrepancies out of 80 Queries)", ha='center', va='center', fontsize=12, fontweight='bold', color='#0f172a')

    # Query 1: TXT_19
    c1 = patches.FancyBboxPatch((0.05, 0.48), 0.42, 0.38, boxstyle="round,pad=0.015",
                                facecolor="#eff6ff", edgecolor="#2563eb", lw=1.5)
    ax.add_patch(c1)
    ax.text(0.26, 0.81, "Case 1: TXT_19 (Evaluator Naming Artifact)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1e3a8a')
    ax.text(0.07, 0.74, "Query: 'Who was the first British woman to win a Nobel Prize?'", fontsize=7.5, color='#0f172a', style='italic')
    ax.text(0.07, 0.67, "Routed Prediction: TXTSRC (99.96% confidence) -> Search-RAG Handler", fontsize=7.5, color='#1e40af')
    ax.text(0.07, 0.60, "Corpus Citation: 'Dorothy Crowfoot Hodgkin' (from REF_19)", fontsize=7.5, color='#15803d')
    ax.text(0.07, 0.53, "Evaluator Criterion: Exact substring 'Dorothy Hodgkin'", fontsize=7.5, color='#b91c1c')
    ax.text(0.26, 0.44, "Classification: NOT a Model Hallucination\n(Metric substring evaluation sensitivity)", ha='center', va='center', fontsize=8, fontweight='bold', color='#1e3a8a')

    # Query 2 & 3: SEM_08 & SEM_17
    c2 = patches.FancyBboxPatch((0.53, 0.48), 0.42, 0.38, boxstyle="round,pad=0.015",
                                facecolor="#fffbeb", edgecolor="#d97706", lw=1.5)
    ax.add_patch(c2)
    ax.text(0.74, 0.81, "Cases 2 & 3: SEM_08 & SEM_17 (Boundary Error)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#92400e')
    ax.text(0.55, 0.74, "SEM_08: 'Write a 4-line rhyming poem (AABB)... about autumn.'", fontsize=7.5, color='#0f172a', style='italic')
    ax.text(0.55, 0.68, "SEM_17: 'Write a creative haiku (5-7-5 syllable)... about sunrise.'", fontsize=7.5, color='#0f172a', style='italic')
    ax.text(0.55, 0.61, "Predicted: CSTRSET (86.5% & 99.9% conf) -> Constrained Solver", fontsize=7.5, color='#b91c1c')
    ax.text(0.55, 0.54, "Root Cause: Structural constraints triggered schema prior", fontsize=7.5, color='#b45309')
    ax.text(0.74, 0.44, "Classification: Classifier Routing Misclassification\n(Structural constraint keyword overriding generative intent)", ha='center', va='center', fontsize=8, fontweight='bold', color='#92400e')

    # Summary bar at bottom
    b_sum = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.28, boxstyle="round,pad=0.015",
                                   facecolor="#f8fafc", edgecolor="#475569", lw=1.2)
    ax.add_patch(b_sum)
    ax.text(0.50, 0.30, "Systemic Insight & Mitigation Architecture", ha='center', va='center', fontsize=10, fontweight='bold', color='#0f172a')
    ax.text(0.50, 0.22, "1. Phase 6.7 boundary training successfully cured 3 of 5 boundary failures (SEM_03, SEM_12, SEM_16).", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(0.50, 0.16, "2. Optional user-provided intent at query time allows end-users to designate SEMADPT directly for constrained creative text.", ha='center', va='center', fontsize=8.5, color='#334155')
    ax.text(0.50, 0.10, "3. Confidence-aware fallback: Ambiguous boundary queries can default to cloud LLM generation to eliminate solver dead-ends.", ha='center', va='center', fontsize=8.5, color='#334155')

    plt.tight_layout()
    plt.savefig('artifacts/figures/fig10_failure_flow.png', dpi=300)
    plt.close()
    print("Saved fig10_failure_flow.png")

if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_6()
    generate_figure_7()
    generate_figure_8()
    generate_figure_9()
    generate_figure_10()
    print("All 10 figures successfully generated.")
