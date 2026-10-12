"""
Academic Literature Review Section and Comprehensive References Builder
Fully source-grounded in the 21 research papers from literature review/
and the project's foundational dataset/model sources.
"""

from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

def add_literature_review_section(doc, add_h1, add_h2, add_h3, add_p, format_table, add_callout):
    # =========================================================================
    # 2. LITERATURE REVIEW
    # =========================================================================
    add_h1("2. Literature Review")
    add_p(
        "The deployment of Large Language Models (LLMs) as universal interfaces for diverse computation and reasoning "
        "tasks has exposed a critical tension between generative flexibility and domain-specific operational guarantees. "
        "While autoregressive foundation models demonstrate remarkable breadth across broad open-ended domains, their "
        "monolithic application to exact arithmetic, closed-world factual lookup, and deterministic schema serialization "
        "incurs severe vulnerabilities to confabulation, substantial latency overhead, and escalating financial expenditures. "
        "To contextualize the intent-routed architecture investigated in this work, this section reviews six foundational bodies "
        "of research: (1) LLM hallucination and reliability limitations, (2) intent classification and query understanding, "
        "(3) retrieval-augmented generation and external non-parametric grounding, (4) constraint-based and structured output generation, "
        "(5) tool-augmented generation and symbolic delegation, and (6) mixture-of-experts, model routing, and system efficiency. "
        "Finally, this synthesis derives the core academic research gap that motivates our investigation of pre-generation "
        "intent-driven routing across heterogeneous computational modalities."
    )

    # -------------------------------------------------------------------------
    # 2.1 LLM Hallucination and Reliability Limitations
    # -------------------------------------------------------------------------
    add_h2("2.1 LLM Hallucination, Factuality, and Reliability Limitations")
    add_p(
        "Autoregressive language models generate tokens sequentially by modeling conditional probability distributions "
        "over vocabularies given a prefix context. Consequently, their outputs are governed by statistical likelihood rather "
        "than formal verification, leaving them inherently susceptible to generating plausible yet factually incorrect or "
        "unsupported statements—a phenomenon widely categorized as hallucination or confabulation [10]–[16]."
    )
    add_p(
        "In a seminal investigation of neural abstractive summarization, Maynez et al. [15] established a rigorous distinction "
        "between faithfulness (whether generated content is strictly entailed by the input source document) and factuality "
        "(whether generated content reflects true real-world facts). Their large-scale human evaluation across extreme summarization "
        "(XSum [4]) revealed that even modern transformer-based seq2seq models produce unfaithful or hallucinated content in over "
        "30% of generated sentences. Crucially, the authors demonstrated that standard n-gram overlap metrics (such as ROUGE) "
        "exhibit near-zero correlation with human judgments of factual correctness, whereas natural language inference (NLI) "
        "and textual entailment models provide significantly stronger alignment with human annotations of hallucination [15]."
    )
    add_p(
        "The vulnerability of foundation models extends across academic and reasoning domains. Hendrycks et al. [11] introduced "
        "the Massive Multitask Language Understanding (MMLU) benchmark spanning 57 diverse subjects, finding that foundation models "
        "exhibit starkly lopsided capabilities: while achieving competitive performance in elementary humanities, they suffer from "
        "near-random accuracy on specialized technical tasks, elementary mathematics, and professional domains. Liang et al. [10] "
        "expanded this diagnosis through the Holistic Evaluation of Language Models (HELM), evaluating foundation models across "
        "42 scenarios and 7 multi-dimensional metrics. HELM demonstrated that conventional evaluation benchmarks focusing solely on "
        "top-line accuracy fail to capture severe deficiencies in model calibration, demographic robustness, copyright memorization, "
        "and factual stability [10]."
    )
    add_p(
        "A critical vulnerability in generative reasoning is context distractibility. Shi et al. [14] investigated the robustness "
        "of state-of-the-art LLMs on the Grade-School Math with Irrelevant Context (GSM-IC) benchmark, demonstrating that prepending "
        "or interleaving irrelevant sentences into math word problems causes catastrophic accuracy drops across both Chain-of-Thought (CoT) "
        "and self-consistency prompting. LLMs frequently misbind extraneous numeric values into their internal reasoning chains, "
        "highlighting the fragility of general-purpose generative architectures when processing inputs without explicit structural scoping [14]."
    )
    add_p(
        "To mitigate hallucination, substantial research has investigated post-generation detection techniques. Kadavath et al. [13] "
        "studied whether language models can evaluate their own validity, showing that larger models exhibit well-calibrated self-probabilities "
        "P(True) on multiple-choice formats, but remain prone to overconfidence on open-ended generation, where reinforcement learning from "
        "human feedback (RLHF) can distort probability calibration. Manakul et al. [16] proposed SelfCheckGPT, a zero-resource black-box "
        "hallucination detection framework based on the hypothesis that when an LLM possesses genuine knowledge of a concept, stochastically "
        "sampled completions will consistently agree, whereas hallucinated facts will diverge and contradict one another across samples. "
        "Farquhar et al. [12] introduced semantic entropy in Nature, presenting an unsupervised method for detecting confabulations by "
        "clustering stochastically sampled generations into semantic equivalence classes using bidirectional NLI. By measuring entropy over "
        "meanings rather than superficial lexical tokens, semantic entropy reliably identifies confabulations across diverse tasks [12]."
    )
    add_p(
        "However, a fundamental architectural limitation shared by both SelfCheckGPT [16] and semantic entropy [12] is their post-hoc, "
        "sampling-based nature. To evaluate a single response, the system must generate N stochastic completions (typically N = 5 to 20), "
        "multiplying end-to-end latency and commercial API costs by N-fold. While effective for offline auditing, post-generation "
        "detection is computationally prohibitive for high-throughput operational deployments, strongly motivating pre-generation "
        "mechanisms that prevent generative failure architecturally before expensive token generation occurs."
    )

    # -------------------------------------------------------------------------
    # 2.2 Intent Classification and Query Understanding
    # -------------------------------------------------------------------------
    add_h2("2.2 Intent Classification and User Query Understanding")
    add_p(
        "User intent classification has long served as a cornerstone of spoken language understanding (SLU) and task-oriented dialogue "
        "systems [2], [17]. In production conversational interfaces, accurately categorizing user utterances into discrete intent classes "
        "enables conversational agents to invoke appropriate back-end fulfillment logic and maintain structured dialogue states."
    )
    add_p(
        "Larson et al. [2] introduced the benchmark CLINC150 dataset, comprising 23,700 queries across 150 distinct intent categories and "
        "10 domains (including banking, travel, work, and utility commands), while formalizing the out-of-scope (OOS) classification challenge. "
        "Their empirical findings established that fine-tuned transformer encoders achieve superior in-scope classification accuracy (exceeding "
        "96%) even under limited training samples and class imbalance. Castellucci et al. [17] developed Bert-Joint, demonstrating that "
        "joint intent detection and slot filling within bidirectional transformer encoder architectures establishes near-saturated accuracy "
        "on standard spoken language benchmarks (such as ATIS and Snips) across multiple languages, validating the expressiveness of deep "
        "bidirectional representations for utterance-level semantic discrimination."
    )
    add_p(
        "The representational capability of transformer encoders was further refined by Liu et al. [18] in RoBERTa, which demonstrated "
        "that dynamic masking, larger mini-batch training, and the removal of the next-sentence prediction objective produce highly robust "
        "contextual representations across the GLUE benchmark. Concurrently, Sanh et al. [6] introduced DistilBERT, demonstrating that "
        "knowledge distillation can compress the 110M parameter BERT architecture into a lightweight 66M parameter model that retains 97% "
        "of full language understanding capabilities while executing inference 60% faster. DistilBERT provides an exceptional balance of "
        "representational richness and microsecond-level execution efficiency, making it an ideal candidate for low-latency pre-generation "
        "gating in real-time pipelines [6]."
    )
    add_p(
        "In parallel, foundation model alignment research has sought to embed intent following directly into generative decoders. Ouyang et al. [19] "
        "introduced InstructGPT, showing that fine-tuning language models with human demonstrations (Supervised Fine-Tuning, SFT) and "
        "reinforcement learning from human feedback (RLHF) aligns model completions with user intent far more effectively than next-token "
        "pre-training alone. Human evaluators overwhelmingly preferred completions from a 1.3B parameter InstructGPT model over those from the "
        "unaligned 175B parameter GPT-3 baseline [19]. Nevertheless, the authors emphasized that instruction alignment does not eradicate factual "
        "hallucinations or arithmetic errors, as the underlying autoregressive decoding mechanism remains probabilistic."
    )

    # -------------------------------------------------------------------------
    # 2.3 Retrieval, External Evidence, and Non-Parametric Grounding
    # -------------------------------------------------------------------------
    add_h2("2.3 Retrieval, External Evidence, and Non-Parametric Grounding")
    add_p(
        "A primary cause of factual hallucination in LLMs is the fundamental boundary of parametric memory. While large models store vast "
        "encyclopedic associations within their feed-forward weights, parametric knowledge is static, costly to update, and incapable of "
        "providing rigorous evidence provenance."
    )
    add_p(
        "Mallen et al. [23] investigated this phenomenon in their study 'When Not to Trust Language Models', evaluating ten foundation models "
        "across the entity-centric PopQA and EntityQuestions benchmarks. By analyzing query popularity based on Wikipedia page views, the authors "
        "uncovered a profound scaling law: parameter scaling significantly enhances LLM memorization of popular 'head' entities, but fails to "
        "appreciably improve factual recall on long-tail knowledge. For long-tail factual inquiries, parametric LLMs consistently hallucinate. "
        "Conversely, augmenting models with non-parametric retrieval dramatically improves factual accuracy across the entire distribution, "
        "proving that factual inquiries cannot rely exclusively on internal parameters [23]."
    )
    add_p(
        "To establish factual grounding, Lewis et al. [22] formulated Retrieval-Augmented Generation (RAG). By combining a pre-trained "
        "parametric seq2seq generator with an explicit non-parametric dense vector index of Wikipedia accessed via Maximum Inner Product "
        "Search (MIPS), RAG models dynamically retrieve supporting document passages conditioned on the input query. In evaluations across "
        "TriviaQA [3], Natural Questions, and CuratedTREC, RAG substantially reduced factual hallucinations and generated more specific, "
        "grounded completions than monolithic parametric baselines, while allowing seamless real-time knowledge base updates without retraining [22]."
    )
    add_p(
        "Similarly, Joshi et al. [3] introduced TriviaQA, providing over 650K question-answer-evidence triples gathered from trivia enthusiasts "
        "and independently collected web documents. Their analysis demonstrated that open-domain factual QA involves complex cross-sentence "
        "dependencies and severe lexical variability between questions and evidence passages, reinforcing the conclusion that robust factual "
        "attribution requires coupling generation to explicit reference evidence rather than unassisted parametric recall [3]."
    )

    # -------------------------------------------------------------------------
    # 2.4 Constraint-Based and Structured Output Generation
    # -------------------------------------------------------------------------
    add_h2("2.4 Constraint-Based and Structured Output Generation")
    add_p(
        "Enterprise production deployments routinely require language models to adhere to rigid structural constraints and deterministic "
        "serialization schemas, such as JSON payloads, YAML configurations, XML structures, or exact character/word boundaries. However, "
        "enforcing structural determinism through autoregressive prompt engineering remains notoriously brittle."
    )
    add_p(
        "Zhou et al. [5] established the Instruction-Following Evaluation (IFEval) benchmark, designed to evaluate the adherence of LLMs "
        "to 25 verifiable 'verifiable instructions' (such as JSON formatting, length constraints, punctuation rules, and keyword inclusion/exclusion) "
        "using objective, deterministic programmatic verification rather than subjective LLM evaluators. Their findings revealed that even "
        "state-of-the-art foundation models (including GPT-4 and PaLM 2) exhibit noticeable failure rates when multiple structural constraints "
        "are combined, frequently introducing conversational preambles, syntax errors, or subtle length violations. HELM evaluations by Liang et al. [10] "
        "similarly documented persistent schema degradation on structured data imputation tasks."
    )
    add_p(
        "These structural deficiencies stem from the mismatch between autoregressive decoding and formal grammar semantics: next-token sampling "
        "possesses no inherent concept of syntax trees or schema validation. In contrast, deterministic schema synthesis engines and programmatic "
        "validators guarantee 100% schema compliance with zero syntax error rates, establishing a compelling case for offloading rigid structural "
        "requests to deterministic solvers rather than generative foundation models [5]."
    )

    # -------------------------------------------------------------------------
    # 2.5 Tool-Augmented Generation and Symbolic Delegation
    # -------------------------------------------------------------------------
    add_h2("2.5 Tool-Augmented Generation and Symbolic Delegation")
    add_p(
        "Parallel to retrieval augmentation, a prominent research paradigm has focused on augmenting language models with external "
        "computational tools and symbolic runtimes to overcome innate deficiencies in arithmetic and multi-step logic [1], [24]–[27]."
    )
    add_p(
        "Cobbe et al. [1] highlighted this limitation through the GSM8K benchmark of 8.5K grade-school math word problems. Their findings "
        "demonstrated that even multi-billion parameter transformer models struggle with multi-step arithmetic reasoning, where a single "
        "arithmetic slip in an intermediate reasoning step inevitably corrupts the final answer. To address this, Gao et al. [24] proposed "
        "Program-Aided Language models (PAL). PAL shifts the role of the LLM entirely: instead of performing calculation, the LLM reads the "
        "natural language problem and generates Python code representing intermediate reasoning steps, while offloading actual execution to a "
        "Python interpreter runtime. On GSM8K, PAL achieved a 15% absolute performance improvement over standard Chain-of-Thought prompting, "
        "demonstrating that delegating arithmetic to deterministic symbolic interpreters completely eliminates arithmetic calculation errors [24]."
    )
    add_p(
        "Yao et al. [25] formalized ReAct (Reasoning and Acting), an interactive framework where language models interleave reasoning traces "
        "('Thoughts') with external tool invocations ('Actions') and environment observations. On multi-hop QA (HotpotQA) and fact verification "
        "(FEVER), ReAct effectively overcame hallucination and cascading error propagation by dynamically querying Wikipedia APIs to ground "
        "its reasoning steps. Similarly, Schick et al. [27] developed Toolformer, demonstrating that LLMs can teach themselves in a self-supervised "
        "manner to invoke external tools—including calculators, search engines, translation systems, and calendars—via inline API tokens, "
        "yielding significant improvements on downstream factual and mathematical benchmarks without compromising core language modeling abilities [27]."
    )
    add_p(
        "At the frontier of neuro-symbolic reasoning, Trinh et al. [26] introduced AlphaGeometry in Nature, solving 25 of 30 International "
        "Mathematical Olympiad (IMO) geometry problems. AlphaGeometry combines a neural language model trained on massive synthetic proofs "
        "with a deterministic symbolic deduction engine (DD+AR). The neural model provides intuitive guidance to propose auxiliary geometric "
        "constructions, while the symbolic engine performs mechanical, forward deduction with formal rigor and zero mathematical hallucination [26]. "
        "Together with open-source symbolic libraries like SymPy [7], these works establish that deterministic symbolic engines represent the "
        "gold standard for exact algebraic and numeric evaluation."
    )

    # -------------------------------------------------------------------------
    # 2.6 Mixture-of-Experts, Model Routing, and System Efficiency
    # -------------------------------------------------------------------------
    add_h2("2.6 Mixture-of-Experts, Model Routing, and System Efficiency")
    add_p(
        "As the operational costs of foundation models have escalated, researchers have increasingly investigated dynamic routing, "
        "model selection, and efficiency optimization to reduce compute budgets while maintaining task accuracy [20], [21]."
    )
    add_p(
        "Shnitzer et al. [20] addressed the challenge of selecting the optimal LLM from a collection of candidate models in their study "
        "'Large Language Model Routing with Benchmark Datasets'. Analyzing 18 foundation models across 29 HELM benchmark tasks, the authors "
        "demonstrated that no single model dominates across all use cases. They formulated model routing as a supervised classification "
        "problem, training lightweight binary classifiers on benchmark evaluation outputs to predict whether a given model would succeed on a "
        "query. Their resulting router consistently outperformed any individual model across diverse tasks while substantially reducing "
        "computational and inference costs [20]."
    )
    add_p(
        "In the parameter-adaptation domain, Hu et al. [21] introduced LoRA (Low-Rank Adaptation), freezing pretrained foundation weights and "
        "injecting trainable low-rank decomposition matrices into transformer attention layers. While LoRA reduces trainable parameters by "
        "10,000x and GPU memory requirements by 3x, deploying specialized adapter weights for distinct tasks still requires running inference "
        "through multi-billion parameter foundation backbones, retaining the baseline's latency and hosting footprint [21]."
    )
    add_p(
        "To clarify architectural terminology, several distinct routing and specialization paradigms must be differentiated:"
        "\n1. Model-Level Routing (Model Selection): Dynamically selecting among an ensemble of foundation LLMs (e.g., routing between GPT-4, "
        "Llama-2, and Claude based on predicted task difficulty, as in Shnitzer et al. [20])."
        "\n2. Parameter-Level Specialization: Injecting task-specific low-rank adapters into a shared generative backbone (e.g., LoRA [21])."
        "\n3. In-Generation Tool Calling: Prompting or fine-tuning an LLM to emit API call syntax during autoregressive token decoding (e.g., "
        "Toolformer [27], ReAct [25], and PAL [24])."
        "\n4. Agentic Multi-Hop Loops: Multi-turn iterative agent frameworks where an LLM repeatedly reflects, acts, and parses observations "
        "across several sequential generative calls (e.g., ReAct [25])."
        "\n5. Pre-Generation Architectural Routing: The approach investigated in this project, where a lightweight discriminative encoder "
        "categorizes user query intent prior to any generative token emission and dispatches the query to fundamentally distinct, heterogeneous "
        "execution modalities—routing non-generative tasks entirely away from foundation LLMs into deterministic local engines."
    )

    # -------------------------------------------------------------------------
    # 2.7 Comparative Summary Table
    # -------------------------------------------------------------------------
    add_h2("2.7 Comparative Summary of Reviewed Literature")
    add_p(
        "Table LR-1 provides a systematic comparison of representative studies across the reviewed literature, summarizing the core problem "
        "addressed, methodology, evaluation benchmarks, primary findings, and key architectural limitations or relevance to the present study."
    )

    table_data = [
        [
            "Maynez et al. (2020) [15]",
            "Faithfulness vs. factuality in abstractive summarization",
            "Human evaluation & NLI textual entailment scoring of seq2seq summarizers",
            "XSum, CNN/DailyMail (N=500 per model)",
            "Over 30% of generated sentences contain hallucinations; ROUGE fails to detect unfaithfulness; NLI correlates strongly",
            "Confirms high hallucination rate in open generation; motivates reference corpus grounding for TXTSRC"
        ],
        [
            "Hendrycks et al. (2021) [11]",
            "Multitask knowledge & academic reasoning breadth",
            "MMLU benchmark across 57 subjects (STEM, humanities, social sciences)",
            "15,908 multiple-choice questions across diverse academic fields",
            "Foundation models exhibit lopsided capabilities; near-random accuracy on elementary math and specialized topics",
            "Demonstrates that monolithic LLMs cannot be assumed to possess universal reliability across task domains"
        ],
        [
            "Cobbe et al. (2021) [1]",
            "Multi-step mathematical reasoning in autoregressive models",
            "GSM8K benchmark & training verifiers to rerank generated candidate solutions",
            "8,500 grade school math word problems",
            "Autoregressive generation suffers frequent intermediate calculation errors; verification scales better than fine-tuning",
            "Highlights inherent calculation fragility; motivates offloading arithmetic to deterministic solvers (MATHDS)"
        ],
        [
            "Lewis et al. (2020) [22]",
            "Factual hallucinations & knowledge staleness in seq2seq models",
            "Retrieval-Augmented Generation (RAG) combining parametric seq2seq with dense Wikipedia index (MIPS)",
            "TriviaQA, Natural Questions, WebQuestions, CuratedTREC",
            "Non-parametric retrieval substantially improves factual accuracy, supports real-time updates, and provides provenance",
            "Direct architectural foundation for our TXTSRC SearchRAGHandler reference corpus retrieval"
        ],
        [
            "Kadavath et al. (2022) [13]",
            "Epistemic calibration & self-evaluation of validity in LLMs",
            "Probing LLM probabilities P(True) and P(IK) across diverse sampling tasks",
            "MMLU, TriviaQA, GSM8K, Math, synthetic QA",
            "Larger models are well-calibrated on multiple choice, but remain overconfident and ungrounded on free generation",
            "Confirms that internal confidence during open-ended generation is insufficient to prevent confabulation"
        ],
        [
            "Farquhar et al. (2024) [12]",
            "Unsupervised detection of confabulations in black-box LLMs",
            "Semantic entropy: clustering stochastic completions via bidirectional NLI entailment",
            "TriviaQA, CoQA, BioASQ, SQuAD across GPT-4, Llama, and Gemini",
            "Semantic clustering outperforms token entropy, accurately detecting confabulations without ground-truth labels",
            "Sampling-based post-hoc detection requires N multiple completions, multiplying latency and cost by N-fold"
        ],
        [
            "Manakul et al. (2023) [16]",
            "Zero-resource black-box hallucination detection",
            "SelfCheckGPT: measuring information divergence across stochastically sampled responses",
            "WikiBio hallucination dataset with GPT-3 generations",
            "Fact-checked responses can be detected without external databases via sample inconsistency",
            "Post-generation sampling incurs heavy API invocation overhead, reinforcing need for pre-generation routing"
        ],
        [
            "Shi et al. (2023) [14]",
            "Context distractibility in arithmetic word problem solving",
            "GSM-IC benchmark introducing irrelevant sentences into math word problems",
            "GSM8K with programmatic irrelevant context additions",
            "Irrelevant context dramatically degrades LLM accuracy across CoT and self-consistency prompting",
            "Proves vulnerability of generative reasoning; supports dedicated intent scoping before problem dispatch"
        ],
        [
            "Gao et al. (2023) [24]",
            "Arithmetic calculation and logic errors in LLM reasoning",
            "Program-Aided Language models (PAL): LLM writes Python code; runtime executes logic",
            "GSM8K, SVAMP, ASDiv, BIG-Bench Hard (13 tasks)",
            "Offloading execution to a Python interpreter improves mathematical reasoning by +15% over CoT",
            "Validates delegating computation to symbolic interpreters, but still requires LLM code generation"
        ],
        [
            "Trinh et al. (2024) [26]",
            "Olympiad-level mathematical theorem proving without human proofs",
            "AlphaGeometry: neuro-symbolic coupling of language model search guidance with symbolic deduction engine (DD+AR)",
            "30 IMO geometry problems (2000–2022)",
            "Solves 25 of 30 problems (approaching IMO gold medallist); produces fully verifiable human-readable proofs",
            "Exemplifies the neuro-symbolic paradigm: neural models guide search, while symbolic engines guarantee truth"
        ],
        [
            "Zhou et al. (2023) [5]",
            "Evaluating strict instruction following and structural constraints",
            "IFEval benchmark with 25 verifiable constraint types (JSON, word count, formatting)",
            "541 prompt instances with programmatic constraint verifiers",
            "State-of-the-art LLMs struggle with multi-constraint formatting, frequently violating structural rules",
            "Motivates CSTRSET ConstrainedSolverHandler: deterministic schema synthesis guarantees 100% compliance"
        ],
        [
            "Shnitzer et al. (2023) [20]",
            "Selecting optimal LLMs across diverse benchmark distributions",
            "Supervised binary router models learned from benchmark evaluation matrices across 18 candidate LLMs",
            "29 HELM benchmarks across reasoning, QA, and summarization",
            "Router outperforms any single foundation model on average while reducing compute cost",
            "Focuses strictly on model-level selection among LLMs; does not route to non-generative local engines"
        ]
    ]

    tbl_lr = doc.add_table(rows=1, cols=6)
    col_widths = [Inches(1.2), Inches(1.1), Inches(1.3), Inches(1.1), Inches(1.4), Inches(1.4)]
    headers = ["Study & Reference", "Problem Addressed", "Core Methodology", "Evaluation Benchmarks", "Key Findings", "Relevance / Limitations"]
    format_table(tbl_lr, col_widths, headers, table_data)

    p_lr_cap = doc.add_paragraph()
    p_lr_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_lr_cap.paragraph_format.space_before = Pt(2)
    p_lr_cap.paragraph_format.space_after = Pt(10)
    run_lr_cap = p_lr_cap.add_run("Table LR-1: Systematic Comparison of Related Literature and System Paradigms Across the Reviewed Corpus.")
    run_lr_cap.font.name = 'Calibri'
    run_lr_cap.font.size = Pt(9.5)
    run_lr_cap.font.italic = True
    run_lr_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # -------------------------------------------------------------------------
    # 2.8 Research Gap and Problem Motivation
    # -------------------------------------------------------------------------
    add_h2("2.8 Research Gap and Problem Motivation")
    add_p(
        "Synthesizing the reviewed literature reveals a fundamental structural dichotomy in current AI system architectures:"
    )
    add_p(
        "1. Inefficiency of Post-Hoc Hallucination Detection: State-of-the-art hallucination mitigation primarily relies on post-generation "
        "stochastic verification (such as SelfCheckGPT [16] and Semantic Entropy [12]) or inline iterative tool loops (such as ReAct [25] and "
        "Toolformer [27]). Crucially, both approaches require every incoming query to traverse a heavyweight generative foundation model, "
        "incurring full token decoding latencies (1.5 to 8.5 seconds) and accumulating substantial commercial API fees. While effective at "
        "identifying confabulations, post-hoc sampling multiplies inference expenditure without preventing the underlying model from generating "
        "unsupported statements in the first place.\n"
        "2. Scope Limitations of Prior Routing Approaches: While model routing research (such as Shnitzer et al. [20]) demonstrates the "
        "benefits of dynamically dispatching queries, previous formulations focus almost exclusively on model-level selection—deciding which "
        "generative LLM within a candidate pool should generate the completion. These systems overlook the massive efficiency, latency, and "
        "determinism advantages of routing queries entirely out of the generative LLM pipeline into local, non-generative deterministic engines "
        "when queries require closed-form arithmetic, corpus-grounded retrieval, or strict schema serialization.\n"
        "3. Absence of Heterogeneous Modality Routing: There is an absence of empirical research investigating pre-generation intent routing "
        "across heterogeneous computational modalities—combining symbolic computer algebra (SymPy [7]), closed-world dense retrieval (RAG [22]), "
        "deterministic schema construction, and generative cloud foundation models (Gemini [8]) under a unified, lightweight discriminative gatekeeper.\n"
        "4. Unquantified Cost–Accuracy–Latency Tradeoff: Most critically, the literature lacks a comprehensive, multi-objective quantification "
        "of the Cost–Accuracy–Latency Pareto tradeoff governing pre-generation routing. Can a compact sequence classifier (such as a 66M parameter "
        "DistilBERT [6]) operating at sub-50ms latency reliably categorize user intent with sufficient precision to eliminate unnecessary API "
        "invocations and slash median latency by an order of magnitude, while maintaining near-parity with direct foundation LLM accuracy? "
        "Furthermore, what representational boundary dilemmas emerge when small discriminative models must differentiate between creative "
        "stylistic adaptation with soft constraints and rigid structural schema constraints?"
    )
    add_p(
        "This research project directly addresses this gap. By constructing IntentFlow, a dual-mode routing framework driven by a fine-tuned "
        "DistilBERT sequence classifier mapped to a four-class operational taxonomy (MATHDS, TXTSRC, SEMADPT, CSTRSET), this study empirically "
        "evaluates the performance, latency, and economic frontiers of pre-generation routing compared against direct foundation LLM generation. "
        "Importantly, our empirical findings do not presuppose that routing universally outperforms monolithic generation in top-line accuracy; "
        "rather, the investigation reveals a nuanced Pareto frontier where dramatic gains in speed, cost, and local computational determinism "
        "are balanced against subtle representational boundaries in natural language intent."
    )
    add_callout(
        doc,
        "Identified Research Gap: Prior literature addresses hallucination through expensive post-generation sampling (multiplying latency "
        "and cost) or restricts routing to model-level selection among alternative generative LLMs. Crucially, the field lacks empirical "
        "evaluation of pre-generation, intent-driven routing across heterogeneous computational modalities (symbolic math, grounded retrieval, "
        "schema solvers, and cloud LLMs) and has not quantified the multi-dimensional Cost–Accuracy–Latency Pareto tradeoff that governs such architectures.",
        title="Formulation of the Core Academic Research Gap",
        border_color="0284C7", bg_color="F0F9FF"
    )

def add_comprehensive_references_section(doc, add_h1, add_p):
    # =========================================================================
    # REFERENCES
    # =========================================================================
    add_h1("References")
    add_p(
        "The following bibliography documents all academic papers, empirical benchmarks, foundation model APIs, and software libraries "
        "reviewed and utilized throughout this study, formatted in standard IEEE citation convention:"
    )

    references_list = [
        "[1] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "
        "\"Training verifiers to solve math word problems,\" arXiv preprint arXiv:2110.14168, 2021.",

        "[2] S. Larson, A. Mahendran, J. J. Peper, C. Clarke, A. Lee, P. Hill, J. K. Kummerfeld, K. Leach, M. A. Laurenzano, L. Tang, and J. Mars, "
        "\"An evaluation dataset for intent classification and out-of-scope prediction,\" in Proceedings of the 2019 Conference on Empirical Methods "
        "in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019, pp. 1311–1316.",

        "[3] M. Joshi, E. Choi, D. S. Weld, and L. Zettlemoyer, "
        "\"TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension,\" in Proceedings of the 55th Annual Meeting of the "
        "Association for Computational Linguistics (ACL), Vancouver, Canada, 2017, pp. 1601–1611.",

        "[4] S. Narayan, S. B. Cohen, and M. Lapata, "
        "\"Don't give me the details, just the summary! Topic-aware convolutional neural networks for extreme summarization,\" in Proceedings of the 2018 "
        "Conference on Empirical Methods in Natural Language Processing (EMNLP), Brussels, Belgium, 2018, pp. 1797–1807.",

        "[5] J. Zhou, T. Müller, S. Shen, Z. Zhou, R. Liu, X. Garcia, X. Zou, and H. Zhao, "
        "\"Instruction-following evaluation for large language models,\" arXiv preprint arXiv:2311.07911, 2023.",

        "[6] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "
        "\"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,\" arXiv preprint arXiv:1910.01108, 2019.",

        "[7] A. Meurer, C. P. Smith, M. Paprocki, O. Čertík, S. B. Kirpichev, M. Rocklin, A. Kumar, S. Ivanov, J. K. Moore, S. Singh, et al., "
        "\"SymPy: symbolic computing in Python,\" PeerJ Computer Science, vol. 3, p. e103, 2017.",

        "[8] Google GenAI SDK, "
        "\"Gemini 3.1 Flash-Lite Foundation Model API,\" Google Cloud Platform, Technical Documentation, 2025.",

        "[9] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, et al., "
        "\"Transformers: State-of-the-art natural language processing,\" in Proceedings of the 2020 Conference on Empirical Methods in Natural Language "
        "Processing: System Demonstrations, 2020, pp. 38–45.",

        "[10] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al., "
        "\"Holistic evaluation of language models,\" Transactions on Machine Learning Research (TMLR), 2023.",

        "[11] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, "
        "\"Measuring massive multitask language understanding,\" in Proceedings of the International Conference on Learning Representations (ICLR), 2021.",

        "[12] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, "
        "\"Detecting hallucinations in large language models using semantic entropy,\" Nature, vol. 630, no. 8017, pp. 625–630, 2024.",

        "[13] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al., "
        "\"Language models (mostly) know what they know,\" arXiv preprint arXiv:2207.05221, 2022.",

        "[14] F. Shi, X. Chen, K. Misra, N. Scales, D. Dohan, E. Chi, N. Schärli, and D. Zhou, "
        "\"Large language models can be easily distracted by irrelevant context,\" in Proceedings of the 40th International Conference on Machine Learning (ICML), "
        "Honolulu, Hawaii, 2023, pp. 31210–31227.",

        "[15] J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, "
        "\"On faithfulness and factuality in abstractive summarization,\" in Proceedings of the 58th Annual Meeting of the Association for Computational "
        "Linguistics (ACL), 2020, pp. 1906–1919.",

        "[16] P. Manakul, A. Liusie, and M. J. F. Gales, "
        "\"SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models,\" in Proceedings of the 2023 Conference on "
        "Empirical Methods in Natural Language Processing (EMNLP), Singapore, 2023, pp. 9004–9017.",

        "[17] G. Castellucci, V. Bellomaria, A. Favalli, and R. Romagnoli, "
        "\"Multi-lingual intent detection and slot filling in a joint BERT-based model,\" arXiv preprint arXiv:1911.08087, 2019.",

        "[18] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "
        "\"RoBERTa: A robustly optimized BERT pretraining approach,\" arXiv preprint arXiv:1907.11692, 2019.",

        "[19] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al., "
        "\"Training language models to follow instructions with human feedback,\" Advances in Neural Information Processing Systems (NeurIPS), vol. 35, "
        "pp. 27730–27744, 2022.",

        "[20] T. Shnitzer, A. Ou, M. Silva, K. Soule, Y. Sun, J. Solomon, N. Thompson, and M. Yurochkin, "
        "\"Large language model routing with benchmark datasets,\" arXiv preprint arXiv:2309.15789, 2023.",

        "[21] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "
        "\"LoRA: Low-rank adaptation of large language models,\" in Proceedings of the International Conference on Learning Representations (ICLR), 2022.",

        "[22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "
        "\"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" Advances in Neural Information Processing Systems (NeurIPS), vol. 33, "
        "pp. 9459–9474, 2020.",

        "[23] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi, "
        "\"When not to trust language models: Investigating effectiveness of parametric and non-parametric memories,\" in Proceedings of the 61st Annual "
        "Meeting of the Association for Computational Linguistics (ACL), vol. 1, Toronto, Canada, 2023, pp. 9802–9822.",

        "[24] L. Gao, A. Madaan, S. Zhou, U. Alon, P. Liu, Y. Yang, J. Callan, and G. Neubig, "
        "\"PAL: Program-aided language models,\" in Proceedings of the 40th International Conference on Machine Learning (ICML), Honolulu, Hawaii, 2023, "
        "pp. 10764–10799.",

        "[25] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, "
        "\"ReAct: Synergizing reasoning and acting in language models,\" in Proceedings of the International Conference on Learning Representations (ICLR), 2023.",

        "[26] T. H. Trinh, Y. Wu, Q. V. Le, H. He, and T. Luong, "
        "\"Solving olympiad geometry without human demonstrations,\" Nature, vol. 625, no. 7995, pp. 476–482, 2024.",

        "[27] T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, "
        "\"Toolformer: Language models can teach themselves to use tools,\" Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023."
    ]

    for ref in references_list:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_before = Pt(2)
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.left_indent = Inches(0.3)
        p_ref.paragraph_format.first_line_indent = Inches(-0.3)
        run_ref = p_ref.add_run(ref)
        run_ref.font.name = 'Calibri'
        run_ref.font.size = Pt(9.5)
        run_ref.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
