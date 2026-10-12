"""
Response Aggregator Module.

Responsible for:
- Packaging handler execution results, intent metadata, confidence score, and latency into a unified response payload.
- Standardizing final response format across all 4 execution paths:
    MATHDS  -> Math Solver
    TXTSRC  -> Search/RAG Handler
    SEMADPT -> LLM Generator
    CSTRSET -> Constrained Solver
"""

from typing import Dict, Any

class ResponseAggregator:
    def format_response(
        self,
        query: str,
        intent: str,
        confidence: float,
        classification_latency_ms: float,
        handler_name: str,
        handler_output: Any
    ) -> Dict[str, Any]:
        """
        Formats and returns the unified response structure required by the pipeline.
        """
        # Extract a clean, readable text representation of response for high-level consumption
        readable_response = ""
        if isinstance(handler_output, dict):
            if "output_text" in handler_output:
                readable_response = handler_output["output_text"]
            elif "formatted_answer" in handler_output:
                readable_response = handler_output["formatted_answer"]
            elif "answer" in handler_output:
                readable_response = handler_output["answer"]
            elif "generated_text" in handler_output:
                readable_response = handler_output["generated_text"]
            elif "raw_json" in handler_output:
                readable_response = handler_output["raw_json"]
            elif "structured_output" in handler_output:
                readable_response = str(handler_output["structured_output"])
            elif "result" in handler_output:
                readable_response = str(handler_output["result"])
            else:
                readable_response = str(handler_output)
        else:
            readable_response = str(handler_output)

        return {
            "query": query,
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "classification_latency_ms": round(float(classification_latency_ms), 2),
            "handler": handler_name,
            "response": readable_response,
            "handler_details": handler_output
        }
