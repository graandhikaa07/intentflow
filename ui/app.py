"""
IntentFlow Web Application - Phase 6.9 Final Interactive Demo.

A lightweight Flask web interface serving the real research pipeline:
User Query -> DistilBERT Classifier -> Query Router -> Specialized Handlers -> Response Aggregator.

Loads the frozen final classifier (artifacts/checkpoints/best_distilbert_final) once on startup.
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Set root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router.query_router import QueryRouter
from dotenv import load_dotenv

# Load local environment (contains GEMINI_API_KEY if present)
load_dotenv(PROJECT_ROOT / ".env")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static")
)

# Global singleton router instance
ROUTER = None
CHECKPOINT_PATH = str(PROJECT_ROOT / "artifacts" / "checkpoints" / "best_distilbert_final")


def get_router():
    """Initializes and returns the singleton QueryRouter instance."""
    global ROUTER
    if ROUTER is None:
        print(f"[*] Initializing IntentFlow with final classifier: {CHECKPOINT_PATH}...")
        ROUTER = QueryRouter(checkpoint_path=CHECKPOINT_PATH)
        print("[*] Model loaded successfully into memory.")
    return ROUTER


@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def get_info():
    """Returns system metadata and architecture details for the About panel."""
    router = get_router()
    metadata_path = Path(CHECKPOINT_PATH) / "final_model_metadata.json"
    model_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                model_metadata = json.load(f)
        except Exception:
            pass

    return jsonify({
        "status": "online",
        "project_name": "IntentFlow",
        "subtitle": "Intent-aware routing for efficient and reliable LLM responses",
        "model_name": "DistilBERT (66M parameters)",
        "checkpoint": "artifacts/checkpoints/best_distilbert_final",
        "taxonomy": [
            {
                "label": "MATHDS",
                "name": "Mathematical & Quantitative Reasoning",
                "handler": "Deterministic SymPy Solver",
                "execution": "Local deterministic execution (zero hallucination)"
            },
            {
                "label": "TXTSRC",
                "name": "Factual Information Retrieval",
                "handler": "Search / Grounded Reference RAG",
                "execution": "Local grounded corpus lookup (cites source evidence)"
            },
            {
                "label": "SEMADPT",
                "name": "Semantic Adaptation & Creative Generation",
                "handler": "Cloud Foundation LLM (Gemini 3.1 Flash-Lite)",
                "execution": "External API generation (temperature=0.2)"
            },
            {
                "label": "CSTRSET",
                "name": "Constraint & Schema Enforcement",
                "handler": "Constrained Schema Solver",
                "execution": "Local schema synthesizer (eliminates format drift)"
            }
        ],
        "metadata": model_metadata
    })


ROUTING_MAP = {
    "MATHDS": "math_solver",
    "TXTSRC": "rag_search",
    "SEMADPT": "llm_generator",
    "CSTRSET": "constrained_solver"
}


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Executes the pipeline for a user query:
    1. If user provided an explicit intent (MATHDS/TXTSRC/SEMADPT/CSTRSET):
       Classifier is SKIPPED. User-provided intent is authoritative.
       Query -> QueryRouter -> Target Handler -> Response Aggregator
    2. If intent is Auto-detect:
       Classifier runs normally.
       Query -> DistilBERT Classifier -> Intent Label -> Router -> Target Handler -> Response Aggregator
    """
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    raw_intent = data.get("intent")

    if not query:
        return jsonify({
            "status": "error",
            "error": "Query cannot be empty."
        }), 400

    try:
        router = get_router()
        start_total = time.perf_counter()

        user_intent = str(raw_intent).strip().upper() if raw_intent else None
        if user_intent in ["AUTO", "AUTO-DETECT", "NONE", ""]:
            user_intent = None

        if user_intent and user_intent in ROUTING_MAP:
            # USER-PROVIDED INTENT MODE:
            # Authoritative user selection: classifier inference is bypassed
            decision_source = "user_provided"
            intent = user_intent
            confidence = None
            clf_latency_ms = 0.0

            handler_name = ROUTING_MAP[intent]
            if handler_name not in router.handlers:
                raise ValueError(f"Unknown handler for intent '{intent}'")

            handler = router.handlers[handler_name]
            start_handler = time.perf_counter()
            handler_output = handler.execute(query)
            handler_latency_ms = (time.perf_counter() - start_handler) * 1000.0
            total_latency_ms = (time.perf_counter() - start_total) * 1000.0

            # Step 4: Graceful Cascade Fallback for user-provided mode
            fallback_triggered = False
            original_handler = handler_name
            fallback_reason = None
            if handler_name != "llm_generator" and isinstance(handler_output, dict):
                if handler_output.get("status") in ["error", "controlled_fallback"]:
                    if "llm_generator" in router.handlers:
                        fallback_triggered = True
                        original_handler = handler_name
                        fallback_reason = (
                            handler_output.get("error")
                            or handler_output.get("answer")
                            or "Query outside deterministic domain"
                        )
                        llm_out = router.handlers["llm_generator"].execute(query)
                        if isinstance(llm_out, dict) and llm_out.get("status") in ["success", "controlled_generation"]:
                            handler_name = "llm_generator"
                            handler_output = llm_out

            pipeline_output = router.aggregator.format_response(
                query=query,
                intent=intent,
                confidence=1.0,
                classification_latency_ms=0.0,
                handler_name=handler_name,
                handler_output=handler_output
            )
            if fallback_triggered:
                pipeline_output["fallback_triggered"] = True
                pipeline_output["original_intent"] = intent
                pipeline_output["original_handler"] = original_handler
                pipeline_output["fallback_reason"] = fallback_reason
        else:
            # AUTO-DETECT MODE:
            # Full canonical pipeline through frozen DistilBERT classifier
            decision_source = "automatic"
            pred_res = router.predict_intent(query)
            intent = pred_res["predicted_label"]
            confidence = float(pred_res["confidence"])
            clf_latency_ms = float(pred_res["latency_ms"])

            pipeline_output = router.route_and_execute(query)
            total_latency_ms = (time.perf_counter() - start_total) * 1000.0
            handler_name = pipeline_output.get("handler", ROUTING_MAP.get(intent, "unknown"))
            handler_latency_ms = max(0.0, total_latency_ms - clf_latency_ms)

        raw_response = pipeline_output.get("response", "")
        handler_details = pipeline_output.get("handler_details", {})
        fallback_triggered = pipeline_output.get("fallback_triggered", False)
        original_intent = pipeline_output.get("original_intent")
        original_handler = pipeline_output.get("original_handler")
        fallback_reason = pipeline_output.get("fallback_reason")

        # Determine execution venue
        is_gemini = False
        execution_type = "Local"
        api_method = "Not used"
        tokens = None
        cost_usd = 0.0

        if isinstance(handler_details, dict):
            method_str = handler_details.get("method", "")
            if "gemini" in method_str.lower() or "cloud_llm" in method_str.lower():
                is_gemini = True
                if fallback_triggered:
                    execution_type = "Gemini API (Auto-Cascade)"
                    orig_disp = {
                        "math_solver": "MathDS",
                        "rag_search": "TxtSrc",
                        "constrained_solver": "CstrSet"
                    }.get(original_handler, original_intent or "Local")
                    api_method = f"Used (gemini-3.1-flash-lite via Cascade from {orig_disp})"
                else:
                    execution_type = "Gemini API"
                    api_method = "Used (gemini-3.1-flash-lite)"
                tokens = {
                    "prompt_tokens": handler_details.get("prompt_tokens", 0),
                    "completion_tokens": handler_details.get("completion_tokens", 0),
                    "total_tokens": handler_details.get("total_tokens", 0)
                }
                cost_usd = handler_details.get("cost_usd", 0.0)
            elif intent == "SEMADPT":
                execution_type = "Local Fallback"
                api_method = "Not available (local fallback used)"

        if isinstance(handler_details, dict) and "execution_latency_ms" in handler_details:
            handler_latency_ms = float(handler_details["execution_latency_ms"])

        # Format user-facing text cleanly
        user_response = raw_response
        handler_status = "success"
        if not fallback_triggered and isinstance(handler_details, dict) and handler_details.get("status") == "error":
            handler_status = "error"
            user_response = "I was unable to process that query. Please try rephrasing your request or providing more details."

        return jsonify({
            "status": "success",
            "query": query,
            "response": user_response,
            "routing": {
                "intent": intent,
                "fallback_triggered": fallback_triggered,
                "original_intent": original_intent,
                "original_handler": original_handler,
                "fallback_reason": fallback_reason,
                "decision_source": decision_source,
                "decision_display": "Automatic" if decision_source == "automatic" else "User provided",
                "classifier_status": "Used (DistilBERT 66M)" if decision_source == "automatic" else "Not used",
                "confidence": round(confidence, 4) if confidence is not None else None,
                "confidence_pct": f"{confidence * 100:.1f}%" if confidence is not None else None,
                "handler_name": handler_name,
                "handler_display_name": {
                    "math_solver": "Math Solver",
                    "rag_search": "Search / Grounded RAG",
                    "llm_generator": "LLM Generator",
                    "constrained_solver": "Constrained Schema Solver"
                }.get(handler_name, handler_name),
                "execution_type": execution_type,
                "gemini_api": api_method,
                "is_gemini": is_gemini,
                "classification_latency_ms": round(clf_latency_ms, 2) if decision_source == "automatic" else 0.0,
                "handler_latency_ms": round(handler_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "tokens": tokens,
                "cost_usd": round(cost_usd, 6) if cost_usd else 0.0,
                "handler_status": handler_status,
                "provenance": handler_details.get("source") if isinstance(handler_details, dict) else None,
                "method": handler_details.get("method") if isinstance(handler_details, dict) else None
            }
        })

    except Exception as e:
        # Graceful error handling without leaking stack traces or credentials
        err_msg = str(e)
        # Sanitize any accidental paths
        sanitized = err_msg.replace(str(PROJECT_ROOT), "[PROJECT_ROOT]")
        return jsonify({
            "status": "error",
            "error": "An error occurred while processing the query through the routing pipeline.",
            "details": sanitized
        }), 500


if __name__ == "__main__":
    # Pre-warm router
    get_router()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" IntentFlow Web Application Active")
    print(f" Local URL: http://127.0.0.1:{port}")
    print(f" Pipeline: DistilBERT -> QueryRouter -> Handlers")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
