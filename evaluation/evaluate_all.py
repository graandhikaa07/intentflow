"""
Comparative Evaluation Orchestrator.

Responsible for:
- Loading the held-out test split (10%).
- Running evaluation for:
  1. Fine-tuned DistilBERT Intent Classifier
  2. Zero-shot LLM Baseline
- Computing identical metrics under identical conditions:
  - Overall accuracy
  - Per-class precision, recall, and F1
  - Confusion matrix
  - Inference latency (ms/query)
  - Token and API cost ($/1k queries)
- Generating comparative benchmark tables and exporting reports to artifacts/reports/comparison_report.md.
"""

from typing import Dict, Any

class SystemEvaluator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path

    def run_comparative_evaluation(self) -> Dict[str, Any]:
        """
        Runs both models on the test set, computes all metrics, and produces the comparison report.
        """
        raise NotImplementedError("Will be implemented in execution phase.")

if __name__ == "__main__":
    evaluator = SystemEvaluator()
    evaluator.run_comparative_evaluation()
