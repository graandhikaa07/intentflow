"""
Validation script for Phase 6.1 Hallucination Evaluation Benchmark and Reference Corpus.
"""

import json
import re

def main():
    # 1. Load eval_queries.json
    with open("data/hallucination_eval/eval_queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 2. Load reference_corpus.json
    with open("data/hallucination_eval/reference_corpus.json", "r", encoding="utf-8") as f:
        refs = json.load(f)

    print("=================================================================")
    print(" PHASE 6.1 BENCHMARK INTEGRITY VALIDATION")
    print("=================================================================")
    print(f"1. Total evaluation queries: {len(queries)}")
    assert len(queries) == 80, f"Expected 80 queries, got {len(queries)}"

    # Class counts
    counts = {}
    for q in queries:
        counts[q["intent"]] = counts.get(q["intent"], 0) + 1
    print(f"2. Class distribution: {counts}")
    assert counts == {"MATHDS": 20, "TXTSRC": 20, "SEMADPT": 20, "CSTRSET": 20}

    # Unique IDs
    ids = [q["query_id"] for q in queries]
    print(f"3. Unique query IDs count: {len(set(ids))} / {len(ids)}")
    assert len(set(ids)) == 80

    # Field completeness check
    required_fields = ["query_id", "intent", "subdomain", "query", "ground_truth", "evaluation_criteria"]
    for q in queries:
        for rf in required_fields:
            assert rf in q, f"Query {q.get('query_id')} missing {rf}"
    print("4. All queries contain required top-level fields: PASS")

    # Intent-specific verification
    ref_ids = {r["reference_id"] for r in refs}
    for q in queries:
        intent = q["intent"]
        gt = q["ground_truth"]
        ec = q["evaluation_criteria"]
        assert "metric" in ec and "description" in ec
        if intent == "MATHDS":
            assert "expected_value" in gt or "expected_roots" in gt
            assert "evaluation_type" in gt
        elif intent == "TXTSRC":
            assert "reference_id" in gt
            assert gt["reference_id"] in ref_ids, f"Reference ID {gt['reference_id']} not found in reference corpus!"
            assert "target_facts" in gt
        elif intent == "SEMADPT":
            assert "task_type" in gt
            assert "evaluation_type" in gt
        elif intent == "CSTRSET":
            assert "expected_format" in gt
            assert "negative_constraints" in gt
    print("5. Intent-specific ground truth and criteria verification: PASS")

    # Reference corpus entry check
    print(f"6. Reference corpus entries: {len(refs)} (exactly 1 per TXTSRC query)")
    assert len(refs) == 20
    for r in refs:
        assert "reference_id" in r and "entity" in r and "passage" in r and "key_facts" in r and "source" in r
    print("   All reference corpus entries contain required fields: PASS")

    # Secret scanner
    json_dump_all = json.dumps(queries) + json.dumps(refs)
    secret_patterns = [r"AIza[0-9A-Za-z-_]{35}", r"sk-[0-9A-Za-z]{20,}"]
    for p in secret_patterns:
        matches = re.findall(p, json_dump_all)
        assert len(matches) == 0, f"Potential secret match found: {matches}"
    print("7. Secret scan completed: Zero API keys or credentials detected.")

    # Leakage & Overlap Check against train, val, and test
    existing_texts = {}
    splits = ["train", "val", "test"]
    for s in splits:
        with open(f"data/processed/{s}.jsonl", "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                item = json.loads(line)
                existing_texts[item["text"].strip().lower()] = (s, idx)

    exact_duplicates = []
    for q in queries:
        norm_q = q["query"].strip().lower()
        if norm_q in existing_texts:
            split_info = existing_texts[norm_q]
            exact_duplicates.append((q["query_id"], q["query"], split_info))

    print(f"8. Exact cross-split duplicate count: {len(exact_duplicates)}")
    assert len(exact_duplicates) == 0, f"Exact duplicates found: {exact_duplicates}"

    # Near-duplicate analysis with train split
    high_sim_pairs = []
    for q in queries:
        q_words = set(re.findall(r"\w+", q["query"].lower()))
        for t, (split, idx) in existing_texts.items():
            t_words = set(re.findall(r"\w+", t))
            if not q_words or not t_words:
                continue
            jaccard = len(q_words & t_words) / len(q_words | t_words)
            if jaccard >= 0.75:
                high_sim_pairs.append((q["query_id"], q["query"], t, jaccard))

    print(f"9. High lexical similarity pairs (Jaccard >= 0.75): {len(high_sim_pairs)}")
    if high_sim_pairs:
        for qid, qtext, ttext, score in high_sim_pairs[:5]:
            print(f"   [{qid}] '{qtext}' vs '{ttext}' (Score: {score:.2f})")
    print("=================================================================")
    print(" BENCHMARK VALIDATION COMPLETED SUCCESSFULLY [PASS]")
    print("=================================================================")

if __name__ == "__main__":
    main()
