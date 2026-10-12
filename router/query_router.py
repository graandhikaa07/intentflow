"""
Query Router Module.

Responsible for:
- Receiving the user query.
- Querying the fine-tuned DistilBERT intent classifier to obtain the predicted intent label and confidence.
- Dispatching the query to the designated tool handler based on exact architectural mappings:
    MATHDS  -> math_solver        (Deterministic Math Engine via SymPy)
    TXTSRC  -> rag_search          (Search-RAG / Grounded Knowledge Retrieval)
    SEMADPT -> llm_generator       (Open-ended Creative & Tone Adaptation)
    CSTRSET -> constrained_solver  (Rule & Schema Constrained Output)
- Passing execution results through the Response Aggregator to return standardized structured output.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.distilbert_classifier import DistilBERTIntentClassifier
from router.handlers.math_solver import MathSolverHandler
from router.handlers.rag_search import SearchRAGHandler
from router.handlers.llm_generator import LLMGeneratorHandler
from router.handlers.constrained_solver import ConstrainedSolverHandler
from router.aggregator import ResponseAggregator

# Canonical intent-to-handler mapping
ROUTING_MAP = {
    "MATHDS": "math_solver",
    "TXTSRC": "rag_search",
    "SEMADPT": "llm_generator",
    "CSTRSET": "constrained_solver"
}

class QueryRouter:
    def __init__(
        self,
        classifier: Optional[DistilBERTIntentClassifier] = None,
        checkpoint_path: str = "artifacts/checkpoints/best_distilbert",
        handlers: Optional[Dict[str, Any]] = None,
        aggregator: Optional[ResponseAggregator] = None
    ):
        # 1. Initialize classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = DistilBERTIntentClassifier(model_name_or_path=checkpoint_path)

        # 2. Initialize handlers
        if handlers is not None:
            self.handlers = handlers
        else:
            self.handlers = {
                "math_solver": MathSolverHandler(),
                "rag_search": SearchRAGHandler(),
                "llm_generator": LLMGeneratorHandler(),
                "constrained_solver": ConstrainedSolverHandler()
            }

        # 3. Initialize aggregator
        self.aggregator = aggregator if aggregator is not None else ResponseAggregator()

    def predict_intent(self, query: str) -> Dict[str, Any]:
        """
        Runs single-query intent classification via DistilBERT.
        Returns predicted label, confidence, and classification latency.
        """
        pred_res = self.classifier.predict(query)
        return {
            "predicted_label": pred_res["predicted_label"],
            "confidence": pred_res["confidence"],
            "latency_ms": pred_res["latency_ms"]
        }

    def route_and_execute(self, query: str, enable_llm_fallback: bool = True) -> Dict[str, Any]:
        """
        Executes the end-to-end pipeline:
        Query -> DistilBERT Classifier -> Intent Label -> Router -> Target Handler -> Response Aggregator

        Architectural Cascade (Fix 1):
        When a deterministic local handler (math_solver, rag_search, or constrained_solver)
        cannot process the query or reports that it is outside its grammar/corpus,
        the router automatically cascades execution to the foundation LLM (Gemini)
        to ensure seamless generative resolution for out-of-distribution prompts.
        """
        # Step 1: Classify query
        clf_result = self.predict_intent(query)
        intent = clf_result["predicted_label"]
        confidence = clf_result["confidence"]
        clf_latency_ms = clf_result["latency_ms"]

        # Step 2: Resolve target handler from routing map
        handler_name = ROUTING_MAP.get(intent)
        if not handler_name or handler_name not in self.handlers:
            raise ValueError(f"Unknown intent or unmapped handler: '{intent}'")

        handler = self.handlers[handler_name]

        # Step 3: Dispatch query to specialized handler
        handler_output = handler.execute(query)

        # Step 4: Graceful Cascade Fallback (Fix 1)
        fallback_triggered = False
        original_handler = handler_name
        fallback_reason = None

        if enable_llm_fallback and handler_name != "llm_generator":
            # Check if local handler reported failure or out-of-corpus/grammar rejection
            is_unhandled = False
            if isinstance(handler_output, dict):
                status = handler_output.get("status")
                if status in ["error", "controlled_fallback"]:
                    is_unhandled = True
                    fallback_reason = (
                        handler_output.get("error")
                        or handler_output.get("answer")
                        or "Query outside deterministic domain"
                    )

            if is_unhandled and "llm_generator" in self.handlers:
                fallback_triggered = True
                print(f"[*] Architectural Cascade: Handler '{handler_name}' returned '{fallback_reason}'. Cascading to LLM Generator (Gemini)...")
                llm_handler = self.handlers["llm_generator"]
                llm_output = llm_handler.execute(query)
                # If LLM returned valid generated text, adopt it
                if isinstance(llm_output, dict) and llm_output.get("status") in ["success", "controlled_generation"]:
                    handler_name = "llm_generator"
                    handler_output = llm_output

        # Step 5: Aggregate into final standardized response
        final_payload = self.aggregator.format_response(
            query=query,
            intent=intent,
            confidence=confidence,
            classification_latency_ms=clf_latency_ms,
            handler_name=handler_name,
            handler_output=handler_output
        )

        if fallback_triggered:
            final_payload["fallback_triggered"] = True
            final_payload["original_intent"] = intent
            final_payload["original_handler"] = original_handler
            final_payload["fallback_reason"] = fallback_reason

        return final_payload
