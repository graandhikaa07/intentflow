import sys, os, re, json, time
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')
import requests
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
corpus_path = os.path.join(PROJECT_ROOT, "data", "hallucination_eval", "reference_corpus.json")

# Load local benchmark corpus
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

corpus_docs = []
for entry in corpus:
    facts_str = " ".join(f"{k} {v}" for k, v in entry.get("key_facts", {}).items())
    doc = f"{entry.get('entity', '')} {entry.get('topic', '')} {entry.get('passage', '')} {facts_str}"
    corpus_docs.append(doc)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
corpus_vectors = vectorizer.fit_transform(corpus_docs)

def search_live_internet(query: str):
    """Searches the live web using DuckDuckGo with Wikipedia fallback."""
    # 1. DuckDuckGo live search
    try:
        results = list(DDGS().text(query, max_results=3))
        if results:
            snippets = []
            sources = []
            for r in results:
                title = r.get("title", "").strip()
                url = r.get("href", "").strip()
                body = r.get("body", "").strip()
                if body and len(body) > 30:
                    snippets.append(f"**{title}**: {body}")
                    sources.append(f"[{title}]({url})")
            if snippets:
                ans_body = "\n\n".join(snippets[:2])
                source_links = " · ".join(sources[:2])
                return {
                    "status": "success",
                    "method": "live_web_search (DuckDuckGo)",
                    "answer": f"{ans_body}\n\n**Verified Web Sources**: {source_links}",
                    "retrieved_evidence": "\n".join(snippets),
                    "source": sources[0] if sources else "Live Web",
                    "grounded": True
                }
    except Exception:
        pass

    # 2. Wikipedia OpenSearch & Summary API
    clean_q = re.sub(r"(?i)^(what\s+is|who\s+is|who\s+was|where\s+is|when\s+did|explain|tell\s+me\s+about)\s+(the\s+)?", "", query).rstrip("?. ")
    for term in [clean_q, query]:
        try:
            open_url = "https://en.wikipedia.org/w/api.php"
            headers = {"User-Agent": "IntentFlowResearch/1.0 (academic knowledge retrieval)"}
            params = {"action": "opensearch", "search": term, "limit": 2, "namespace": 0, "format": "json"}
            r = requests.get(open_url, params=params, headers=headers, timeout=4).json()
            if len(r) >= 4 and r[1]:
                title = r[1][0]
                wiki_url = r[3][0]
                sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
                sr = requests.get(sum_url, headers=headers, timeout=4).json()
                extract = sr.get("extract")
                if extract and len(extract) > 40:
                    return {
                        "status": "success",
                        "method": "live_web_search (Wikipedia)",
                        "answer": f"According to **Wikipedia** ([{title}]({wiki_url})):\n\n{extract}\n\n**Source**: [{title}]({wiki_url})",
                        "retrieved_evidence": extract,
                        "source": f"Wikipedia ({wiki_url})",
                        "grounded": True
                    }
        except Exception:
            pass

    return None

def hybrid_rag_execute(query: str):
    start = time.perf_counter()
    # Step 1: Check local benchmark corpus
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, corpus_vectors)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])

    if best_sim >= 0.25:
        entry = corpus[best_idx]
        elapsed = (time.perf_counter() - start) * 1000.0
        return {
            "status": "success",
            "method": "grounded_reference_rag (Local Benchmark)",
            "answer": f"According to {entry.get('source')} [{entry.get('reference_id')}]: {entry.get('passage')}",
            "source": entry.get("source"),
            "execution_latency_ms": round(elapsed, 2)
        }

    # Step 2: Live Internet Search
    web_res = search_live_internet(query)
    elapsed = (time.perf_counter() - start) * 1000.0
    if web_res:
        web_res["execution_latency_ms"] = round(elapsed, 2)
        return web_res

    # Step 3: Controlled Fallback if offline / unhandled
    return {
        "status": "controlled_fallback",
        "method": "rag_search",
        "answer": "[Controlled Fallback] Query is outside reference corpus and live web search returned no verified matches.",
        "execution_latency_ms": round(elapsed, 2)
    }

test_queries = [
    "What is the capital of Australia?",
    "What is the speed of light?",
    "What is photosynthesis?",
    "Who is the CEO of Apple?",
    "What is the capital of France?"
]

for q in test_queries:
    res = hybrid_rag_execute(q)
    print("=" * 60)
    print("Q:", q)
    print("Status:", res.get("status"))
    print("Method:", res.get("method"))
    print("Latency:", res.get("execution_latency_ms"), "ms")
    print("Answer Snippet:\n", str(res.get("answer"))[:200] + "...")
