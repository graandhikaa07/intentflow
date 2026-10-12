"""
DistilBERT Intent Classifier Training & Evaluation Pipeline.

Responsible for:
- Pre-training sanity check (model load, forward pass, loss calculation, backward pass).
- Reproducible training loop with AdamW optimizer, warmup scheduling, and cross-entropy loss.
- Epoch-wise validation and checkpointing of the best model to artifacts/checkpoints/best_distilbert.
- Tracking training loss, validation loss, validation accuracy, and execution runtime.
- Post-training held-out test set evaluation:
  Accuracy, Macro/Weighted Precision, Recall, F1, Per-Class metrics, and Confusion Matrix.
- Exporting results to artifacts/reports/.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Support UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure workspace root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from data.dataset_loader import IntentDatasetLoader, IntentDataset
from models.distilbert_classifier import DistilBERTIntentClassifier

def set_seed(seed: int = 42):
    """Sets random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ClassifierTrainer:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        label_map_path: str = "configs/label_map.json",
        checkpoint_subdir: str = "best_distilbert_phase5_1",
        report_prefix: str = "distilbert_phase5_1"
    ):
        self.config_path = config_path
        self.label_map_path = label_map_path
        self.checkpoint_subdir = checkpoint_subdir
        self.report_prefix = report_prefix
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map_data = json.load(f)

        self.seed = self.config.get("seed", 42)
        set_seed(self.seed)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Training device resolved to: {self.device}")

        # Model and training hyperparameters
        self.model_name = self.config.get("model", {}).get("pretrained_model_name", "distilbert-base-uncased")
        self.num_labels = self.config.get("model", {}).get("num_labels", 4)
        self.learning_rate = float(self.config.get("model", {}).get("learning_rate", 2e-5))
        self.num_epochs = int(self.config.get("model", {}).get("num_epochs", 4))
        self.weight_decay = float(self.config.get("model", {}).get("weight_decay", 0.01))
        self.warmup_ratio = float(self.config.get("model", {}).get("warmup_ratio", 0.1))
        
        self.batch_size = int(self.config.get("dataset", {}).get("batch_size", 32))
        self.max_seq_length = int(self.config.get("dataset", {}).get("max_seq_length", 128))

        self.checkpoint_dir = self.config.get("model", {}).get("checkpoint_dir", "artifacts/checkpoints")
        self.best_checkpoint_dir = os.path.join(self.checkpoint_dir, self.checkpoint_subdir)
        self.reports_dir = self.config.get("evaluation", {}).get("output_dir", "artifacts/reports")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_checkpoint_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.label2id = self.label_map_data["label2id"]
        self.id2label = {int(k): v for k, v in self.label_map_data["id2label"].items()}
        self.class_names = [self.id2label[i] for i in range(self.num_labels)]

        # Initialize dataset loader & tokenizer
        self.dataset_loader = IntentDatasetLoader(config_path=config_path, label_map_path=label_map_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def run_sanity_check(self) -> bool:
        """
        Validates model initialization, tokenization, forward pass,
        loss computation, and backward gradient flow before full training.
        """
        print("\n" + "=" * 65)
        print(" [SANITY CHECK] Verifying Pipeline Components...")
        print("=" * 65)

        try:
            # 1. Verify tokenizer & dummy input
            sample_texts = [
                "Calculate the derivative of 3x^2 + 5x - 8",
                "What is the capital of Kazakhstan?"
            ]
            sample_labels = [0, 1]  # MATHDS, TXTSRC

            encodings = self.tokenizer(
                sample_texts,
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            target_labels = torch.tensor(sample_labels, dtype=torch.long).to(self.device)

            print(" [OK] Tokenizer and batch encoding verified.")

            # 2. Initialize test model instance
            test_config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
            test_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=test_config
            ).to(self.device)
            test_model.train()
            print(" [OK] Model weights loaded successfully.")

            # 3. Forward pass & loss
            outputs = test_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_labels
            )
            loss = outputs.loss
            logits = outputs.logits

            assert loss is not None and loss.item() > 0, "Forward pass failed: Loss is None or <= 0"
            assert logits.shape == (2, self.num_labels), f"Expected logits shape (2, {self.num_labels}), got {logits.shape}"
            print(f" [OK] Forward pass verified. Dummy batch loss: {loss.item():.4f}")

            # 4. Backward pass & gradient check
            loss.backward()
            grads_found = False
            for name, param in test_model.named_parameters():
                if "classifier" in name or "pre_classifier" in name:
                    if param.grad is not None and torch.norm(param.grad).item() > 0:
                        grads_found = True
                        break

            assert grads_found, "Backward pass failed: No gradients computed for classification head."
            print(" [OK] Backward pass & gradient computation verified.")
            print("=" * 65)
            print(" [SANITY CHECK] All sanity checks passed! Proceeding to training.\n")
            
            # Clean up test model
            del test_model, outputs, loss, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True

        except Exception as e:
            print(f"\n [FAIL] Sanity check failed with error: {e}")
            raise e

    def train(self) -> Dict[str, Any]:
        """
        Executes the full training and validation routine, saving the best checkpoint.
        """
        # Step 1: Run pre-flight sanity check
        self.run_sanity_check()

        # Step 2: Initialize real model for training
        print(f"[*] Initializing model '{self.model_name}' for 4-class fine-tuning...")
        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)

        # Step 3: Load train and validation dataloaders
        train_texts, train_labels = self.dataset_loader.load_processed_split("train")
        val_texts, val_labels = self.dataset_loader.load_processed_split("val")

        train_dataset = IntentDataset(train_texts, train_labels, self.tokenizer, self.max_seq_length)
        val_dataset = IntentDataset(val_texts, val_labels, self.tokenizer, self.max_seq_length)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Step 4: Setup AdamW Optimizer and Linear Warmup Scheduler
        total_training_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_training_steps * self.warmup_ratio)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
        loss_fn = nn.CrossEntropyLoss()

        print(f"[*] Training configuration:")
        print(f"    - Epochs               : {self.num_epochs}")
        print(f"    - Batch Size           : {self.batch_size}")
        print(f"    - Learning Rate        : {self.learning_rate}")
        print(f"    - Warmup Steps         : {warmup_steps} / {total_training_steps}")
        print(f"    - Train Samples        : {len(train_dataset)}")
        print(f"    - Validation Samples   : {len(val_dataset)}")
        print(f"    - Device               : {self.device}\n")

        # Step 5: Training Loop
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch_duration_sec": []
        }

        best_val_accuracy = -1.0
        best_epoch = -1
        training_start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0.0

            for step, batch in enumerate(train_loader, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            model.eval()
            total_val_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    total_val_loss += loss.item()

                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_targets, val_preds)
            epoch_duration = time.time() - epoch_start

            history["epoch"].append(epoch)
            history["train_loss"].append(round(avg_train_loss, 4))
            history["val_loss"].append(round(avg_val_loss, 4))
            history["val_accuracy"].append(round(val_accuracy, 4))
            history["epoch_duration_sec"].append(round(epoch_duration, 2))

            print(f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Accuracy: {val_accuracy * 100:.2f}% | "
                  f"Time: {epoch_duration:.1f}s")

            # Save checkpoint if best validation accuracy achieved
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                print(f"  [BEST] New best validation accuracy: {val_accuracy * 100:.2f}%. Saving checkpoint to {self.best_checkpoint_dir}...")
                model.save_pretrained(self.best_checkpoint_dir)
                self.tokenizer.save_pretrained(self.best_checkpoint_dir)

        total_training_time = time.time() - training_start_time
        print(f"\n[*] Training complete in {total_training_time:.1f}s ({total_training_time / 60:.2f} min).")
        print(f"[*] Best Model: Epoch {best_epoch} with Val Accuracy: {best_val_accuracy * 100:.2f}%\n")

        training_summary = {
            "num_epochs": self.num_epochs,
            "best_epoch": best_epoch,
            "best_val_accuracy": float(best_val_accuracy),
            "total_training_time_sec": round(total_training_time, 2),
            "history": history,
            "checkpoint_path": self.best_checkpoint_dir
        }

        # Step 6: Run final test set evaluation using the best checkpoint
        test_metrics = self.evaluate_checkpoint(self.best_checkpoint_dir)
        training_summary["test_evaluation"] = test_metrics

        return training_summary

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Loads the best fine-tuned model checkpoint and computes complete metrics
        on the held-out test split (240 samples: 60 per class).
        """
        print("=" * 65)
        print(" [EVALUATION] Evaluating Best Checkpoint on Test Set...")
        print("=" * 65)

        classifier = DistilBERTIntentClassifier(
            model_name_or_path=checkpoint_path,
            label_map_path=self.label_map_path,
            device=str(self.device)
        )

        test_texts, test_labels = self.dataset_loader.load_processed_split("test")
        
        start_time = time.perf_counter()
        predictions = classifier.batch_predict(test_texts, max_seq_length=self.max_seq_length, batch_size=self.batch_size)
        total_eval_time = (time.perf_counter() - start_time) * 1000.0

        y_true = test_labels
        y_pred = [self.label2id[p["predicted_label"]] for p in predictions]
        latencies = [p["latency_ms"] for p in predictions]

        # Metric computations
        acc = accuracy_score(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=list(range(self.num_labels)), zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))

        per_class_summary = {}
        for idx, cls_name in enumerate(self.class_names):
            per_class_summary[cls_name] = {
                "precision": round(float(precision_per_class[idx]), 4),
                "recall": round(float(recall_per_class[idx]), 4),
                "f1": round(float(f1_per_class[idx]), 4),
                "support": int(support_per_class[idx])
            }

        eval_report = {
            "model": "Fine-Tuned DistilBERT (distilbert-base-uncased)",
            "test_sample_count": len(test_texts),
            "test_accuracy": round(float(acc), 4),
            "macro_avg": {
                "precision": round(float(precision_macro), 4),
                "recall": round(float(recall_macro), 4),
                "f1": round(float(f1_macro), 4)
            },
            "weighted_avg": {
                "precision": round(float(precision_weighted), 4),
                "recall": round(float(recall_weighted), 4),
                "f1": round(float(f1_weighted), 4)
            },
            "per_class": per_class_summary,
            "confusion_matrix": {
                "labels": self.class_names,
                "matrix": cm.tolist()
            },
            "latency": {
                "mean_ms": round(float(np.mean(latencies)), 2),
                "p50_ms": round(float(np.percentile(latencies, 50)), 2),
                "p95_ms": round(float(np.percentile(latencies, 95)), 2),
                "total_inference_time_ms": round(total_eval_time, 2)
            }
        }

        # Print human-readable report
        self._print_eval_report(eval_report)

        # Save to artifacts/reports/
        json_report_path = os.path.join(self.reports_dir, f"{self.report_prefix}_test_metrics.json")
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        md_report_path = os.path.join(self.reports_dir, f"{self.report_prefix}_test_report.md")
        self._save_markdown_report(eval_report, md_report_path)

        print(f"[*] Test metrics successfully exported to:")
        print(f"    - JSON : {json_report_path}")
        print(f"    - MD   : {md_report_path}\n")

        return eval_report

    def _print_eval_report(self, r: Dict[str, Any]):
        print("\n" + "=" * 65)
        print(" PHASE 2: DISTILBERT TEST SET PERFORMANCE REPORT")
        print("=" * 65)
        print(f"Model Architecture  : {r['model']}")
        print(f"Test Set Size       : {r['test_sample_count']} samples (60 per class)")
        print(f"Overall Accuracy    : {r['test_accuracy'] * 100:.2f}%")
        print(f"Macro Precision     : {r['macro_avg']['precision'] * 100:.2f}%")
        print(f"Macro Recall        : {r['macro_avg']['recall'] * 100:.2f}%")
        print(f"Macro F1-Score      : {r['macro_avg']['f1'] * 100:.2f}%")
        print(f"Weighted F1-Score   : {r['weighted_avg']['f1'] * 100:.2f}%")
        print(f"Mean Latency        : {r['latency']['mean_ms']} ms/query (P95: {r['latency']['p95_ms']} ms)")

        print("\nPer-Class Classification Metrics:")
        header = f"  {'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cls_name, m in r["per_class"].items():
            print(f"  {cls_name:<10} | {m['precision']*100:>9.2f}% | {m['recall']*100:>9.2f}% | {m['f1']*100:>9.2f}% | {m['support']:>8}")

        print("\nConfusion Matrix (Rows: Ground Truth, Cols: Predicted):")
        labels = r["confusion_matrix"]["labels"]
        cm_matrix = r["confusion_matrix"]["matrix"]
        cm_header = f"  {'':<10} | " + " | ".join(f"{l:<8}" for l in labels)
        print(cm_header)
        print("  " + "-" * (len(cm_header) - 2))
        for i, row in enumerate(cm_matrix):
            row_str = " | ".join(f"{val:>8}" for val in row)
            print(f"  {labels[i]:<10} | {row_str}")
        print("=" * 65 + "\n")

    def _save_markdown_report(self, r: Dict[str, Any], output_path: str):
        cm_labels = r["confusion_matrix"]["labels"]
        cm_matrix = r["confusion_matrix"]["matrix"]

        cm_md_header = "| Ground Truth \\ Predicted | " + " | ".join(cm_labels) + " |"
        cm_md_divider = "| :--- | " + " | ".join([":---:" for _ in cm_labels]) + " |"
        cm_rows = []
        for i, row in enumerate(cm_matrix):
            cm_rows.append(f"| **{cm_labels[i]}** | " + " | ".join(map(str, row)) + " |")

        md_content = f"""# DistilBERT Intent Classifier: Test Evaluation Report

## Executive Summary
- **Model**: `{r['model']}`
- **Test Set Size**: `{r['test_sample_count']}` samples (balanced across 4 classes)
- **Overall Accuracy**: **{r['test_accuracy'] * 100:.2f}%**
- **Macro F1-Score**: **{r['macro_avg']['f1'] * 100:.2f}%**
- **Weighted F1-Score**: **{r['weighted_avg']['f1'] * 100:.2f}%**
- **Inference Latency**: Mean `{r['latency']['mean_ms']} ms` (P50: `{r['latency']['p50_ms']} ms`, P95: `{r['latency']['p95_ms']} ms`)

---

## Per-Class Metrics

| Intent Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
        for cls_name, m in r["per_class"].items():
            md_content += f"| **`{cls_name}`** | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['support']} |\n"

        md_content += f"""
---

## Confusion Matrix

{cm_md_header}
{cm_md_divider}
""" + "\n".join(cm_rows) + f"""

---

## Latency Profile
- **Total Test Inference Time**: {r['latency']['total_inference_time_ms']:.2f} ms
- **Mean Query Latency**: {r['latency']['mean_ms']:.2f} ms
- **P50 (Median) Latency**: {r['latency']['p50_ms']:.2f} ms
- **P95 Latency**: {r['latency']['p95_ms']:.2f} ms
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DistilBERT Intent Classifier")
    parser.add_argument("--checkpoint_subdir", type=str, default="best_distilbert_phase5_1", help="Subdirectory under checkpoints to save best model")
    parser.add_argument("--report_prefix", type=str, default="distilbert_phase5_1", help="Prefix for metrics and markdown report")
    args = parser.parse_args()

    trainer = ClassifierTrainer(
        checkpoint_subdir=args.checkpoint_subdir,
        report_prefix=args.report_prefix
    )
    trainer.train()
