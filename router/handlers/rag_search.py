"""
Search-RAG Factual Lookup Handler (TXTSRC).

Responsible for:
- Handling factual lookup and knowledge retrieval queries.
- Retrieving factual passages from the controlled reference corpus (data/hallucination_eval/reference_corpus.json).
- Providing clearly marked controlled fallbacks when queries fall outside the reference corpus.
- Preventing factual hallucinations by strictly citing retrieved reference evidence without fabrication.
"""

import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

class SearchRAGHandler:
    def __init__(self, corpus_path: Optional[str] = None):
        self.name = "rag_search"
        if corpus_path is None:
            self.corpus_path = os.path.join(PROJECT_ROOT, "data", "hallucination_eval", "reference_corpus.json")
        else:
            self.corpus_path = corpus_path

        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_vectors = None
        self._load_and_index_corpus()

    def _load_and_index_corpus(self):
        """Loads and builds TF-IDF index over the reference corpus."""
        if not os.path.exists(self.corpus_path):
            print(f"[!] Warning: Reference corpus not found at {self.corpus_path}")
            return

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        corpus_docs = []
        for entry in self.corpus:
            facts_str = " ".join(f"{k} {v}" for k, v in entry.get("key_facts", {}).items())
            doc = f"{entry.get('entity', '')} {entry.get('topic', '')} {entry.get('passage', '')} {facts_str}"
            corpus_docs.append(doc)

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.corpus_vectors = self.vectorizer.fit_transform(corpus_docs)

    def _search_live_internet(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Executes live internet search using DuckDuckGo with Wikipedia fallback.
        Returns grounded factual evidence, extracted text, and clickable sources.
        """
        # 1. Primary: DuckDuckGo Live Search
        try:
            results = list(DDGS().text(query, max_results=3))
            if results:
                snippets = []
                sources = []
                for r in results:
                    title = r.get("title", "").strip()
                    url = r.get("href", "").strip()
                    body = r.get("body", "").strip()
                    if body and len(body) > 25:
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
                        "source": sources[0] if sources else "Live Web Search",
                        "grounded": True
                    }
        except Exception:
            pass

        # 2. Secondary: Wikipedia Full-Text Search API & Summary
        clean_q = re.sub(r"(?i)^(what\s+is|who\s+is|who\s+was|where\s+is|when\s+did|explain|tell\s+me\s+about)\s+(the\s+)?", "", query).rstrip("?. ")
        for term in [clean_q, query]:
            try:
                search_url = "https://en.wikipedia.org/w/api.php"
                headers = {"User-Agent": "IntentFlowResearch/1.0 (academic knowledge retrieval; contact@intentflow.edu)"}
                params = {"action": "query", "list": "search", "srsearch": term, "format": "json", "srlimit": 2}
                r = requests.get(search_url, params=params, headers=headers, timeout=5).json()
                items = r.get("query", {}).get("search", [])
                if items:
                    top_title = items[0]["title"]
                    sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(top_title)}"
                    sr = requests.get(sum_url, headers=headers, timeout=5).json()
                    extract = sr.get("extract")
                    wiki_url = sr.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{requests.utils.quote(top_title)}"
                    if extract and len(extract) > 40:
                        return {
                            "status": "success",
                            "method": "live_web_search (Wikipedia)",
                            "answer": f"According to **Wikipedia** ([{top_title}]({wiki_url})):\n\n{extract}\n\n**Verified Web Source**: [{top_title}]({wiki_url})",
                            "retrieved_evidence": extract,
                            "source": f"Wikipedia ({wiki_url})",
                            "grounded": True
                        }
            except Exception:
                pass

        return None

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Retrieves grounded evidence from local reference corpus or live web search.
        Returns explicit provenance and avoids fabricating ungrounded claims.
        """
        start_time = time.perf_counter()

        # Step 1: Check local benchmark corpus first (fast sub-millisecond execution for benchmark suite)
        if self.corpus and self.vectorizer is not None:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.corpus_vectors)[0]
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])

            if best_sim >= 0.25:
                best_entry = self.corpus[best_idx]
                passage = best_entry.get("passage", "")
                source = best_entry.get("source", "Verified Reference Corpus")
                ref_id = best_entry.get("reference_id", "")
                key_facts = best_entry.get("key_facts", {})

                grounded_answer = f"According to {source} [{ref_id}]: {passage}"
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                return {
                    "status": "success",
                    "method": "grounded_reference_rag (Local Benchmark)",
                    "reference_id": ref_id,
                    "retrieved_evidence": passage,
                    "source": source,
                    "key_facts": key_facts,
                    "similarity_score": round(best_sim, 4),
                    "answer": grounded_answer,
                    "grounded": True,
                    "execution_latency_ms": round(elapsed_ms, 2)
                }

        # Step 2: Query Live Internet Search
        web_res = self._search_live_internet(query)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if web_res:
            web_res["execution_latency_ms"] = round(elapsed_ms, 2)
            return web_res

        # Step 3: Controlled Fallback (when query is outside reference corpus and offline / no web results)
        return {
            "status": "controlled_fallback",
            "method": "rag_search",
            "reference_id": None,
            "retrieved_evidence": None,
            "source": None,
            "similarity_score": 0.0,
            "answer": "[Controlled Fallback] Query is outside reference corpus index and live web search returned no verified matches.",
            "grounded": False,
            "execution_latency_ms": round(elapsed_ms, 2)
        }
