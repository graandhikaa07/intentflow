"""
Phase 6.7: SEMADPT/CSTRSET Boundary-Fix DistilBERT Training & Evaluation Pipeline

Trains a new DistilBERT sequence classification model on the V3 Boundary-Augmented
Dataset (26,492 records: 21,193 train, 2,649 val, 2,650 test).

Core Hyperparameters (preserved for comparability):
- Model: distilbert-base-uncased (loaded from models/distilbert-base-uncased)
- Classes: 4 (MATHDS: 0, TXTSRC: 1, SEMADPT: 2, CSTRSET: 3)
- Optimizer: AdamW (weight_decay=0.01, no decay on bias/LayerNorm)
- Learning Rate: 2e-5
- Warmup Ratio: 0.1 (Linear warmup + decay)
- Loss Function: CrossEntropyLoss
- Batch Size: 32
- Epochs: 4
- Gradient Clipping: max_norm=1.0
- Random Seed: 42
- Hardware: CPU (10-12 threads) / CUDA auto-detection
- Checkpoint Directory: artifacts/checkpoints/best_distilbert_phase6_7_boundary/

Evaluations:
1. Held-out V3 Test Set (2,650 samples): Accuracy, Macro/Weighted F1, Per-Class metrics, Confusion Matrix, Latency.
2. Dedicated Phase 6.7 Boundary Evaluation Set (200 unseen samples): Overall accuracy, SEMADPT/CSTRSET P/R/F1, Confusion Matrix, Latency.
3. Untouched 80-Query Benchmark (eval_queries.json): Classifier-only inference, accuracy, analysis of SEM_03, SEM_08, SEM_12, SEM_16, SEM_17, and check of TXT_20 & SEM_20.
"""

import os
import sys
import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

RANDOM_SEED = 42
TARGET_CLASSES = ["MATHDS", "TXTSRC", "SEMADPT", "CSTRSET"]
LABEL2ID = {"MATHDS": 0, "TXTSRC": 1, "SEMADPT": 2, "CSTRSET": 3}
ID2LABEL = {0: "MATHDS", 1: "TXTSRC", 2: "SEMADPT", 3: "CSTRSET"}

MODEL_NAME = str(PROJECT_ROOT / "models" / "distilbert-base-uncased")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "final_intent_dataset_v3_boundary"
CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_distilbert_phase6_7_boundary"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"
BENCHMARK_FILE = PROJECT_ROOT / "data" / "hallucination_eval" / "eval_queries.json"
CORPUS_FILE = PROJECT_ROOT / "data" / "hallucination_eval" / "reference_corpus.json"
HOLDOUT_BOUNDARY_FILE = PROJECT_ROOT / "data" / "hallucination_eval" / "phase6_7_boundary_eval.json"
PROGRESS_FILE = REPORTS_DIR / "phase6_7_training_progress.json"

EXPECTED_BENCHMARK_HASH = "83ccec72b5a0c969e77e36eb00296e1c6e18f9369751038b80a829444f8ee454"
EXPECTED_CORPUS_HASH = "0b390fdb556ac1eecc3b51f37cdfa8d31549b068c824f340b2507e231d161178"
EXPECTED_HOLDOUT_HASH = "401eac2c89763dc87375019a2f8bb71dc220667805933af0034b77da1f27821f"

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

# -----------------------------------------------------------------------------
# Seed & Environment
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_file_hash(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def verify_file_hash(path: Path, expected_hash: str, name: str):
    h = compute_file_hash(path)
    if h != expected_hash:
        raise ValueError(f"FATAL: Hash mismatch for {name}! Found {h}, expected {expected_hash}")
    print(f"[*] Integrity verified: {name} (SHA-256: {h[:16]}...)")

# -----------------------------------------------------------------------------
# Dataset & DataLoader
# -----------------------------------------------------------------------------

class JsonlIntentDataset(Dataset):
    def __init__(self, file_path: Path):
        self.records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line.strip())
                self.records.append({
                    "text": d["text"],
                    "label": LABEL2ID[d["label"]],
                    "label_name": d["label"]
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

def create_collate_fn(tokenizer, max_length=128):
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encodings["labels"] = torch.tensor(labels, dtype=torch.long)
        return encodings
    return collate_fn

# -----------------------------------------------------------------------------
# Sanity Check
# -----------------------------------------------------------------------------

def run_preflight_sanity_check(model_name: str, device: torch.device):
    print("\n" + "=" * 70)
    print(" [SANITY CHECK] Verifying Pretrained Model & Backward Flow...")
    print("=" * 70)
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
    model.train()

    dummy_texts = ["Solve 4x + 7 = 19", "Return exactly 3 JSON fields"]
    dummy_labels = torch.tensor([0, 3]).to(device)
    enc = tokenizer(dummy_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    outputs = model(**enc, labels=dummy_labels)
    assert outputs.loss is not None and outputs.loss.item() > 0
    outputs.loss.backward()

    grads_ok = any(p.grad is not None and torch.norm(p.grad).item() > 0 for n, p in model.named_parameters() if "classifier" in n)
    assert grads_ok, "No gradients in classification head during sanity check!"
    print(f" [OK] Sanity check passed! Initial loss: {outputs.loss.item():.4f}")
    del model, outputs, enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Training Routine
# -----------------------------------------------------------------------------

def train_phase6_7():
    print("=" * 70)
    print("PHASE 6.7: SEMADPT/CSTRSET BOUNDARY-FIX TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Pre-training Benchmark Integrity Verification
    verify_file_hash(BENCHMARK_FILE, EXPECTED_BENCHMARK_HASH, "eval_queries.json")
    verify_file_hash(CORPUS_FILE, EXPECTED_CORPUS_HASH, "reference_corpus.json")
    verify_file_hash(HOLDOUT_BOUNDARY_FILE, EXPECTED_HOLDOUT_HASH, "phase6_7_boundary_eval.json")

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training device: {device}")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 10)
        print(f"[*] Torch CPU threads: {torch.get_num_threads()} (logical cores: {os.cpu_count()})")

    # 2. Pre-flight sanity check
    run_preflight_sanity_check(MODEL_NAME, device)

    # 3. Load V3 Datasets
    print("\n[*] Loading Phase 6.7 V3 boundary dataset splits from:", DATA_DIR)
    train_dataset = JsonlIntentDataset(DATA_DIR / "train.jsonl")
    val_dataset = JsonlIntentDataset(DATA_DIR / "val.jsonl")
    test_dataset = JsonlIntentDataset(DATA_DIR / "test.jsonl")

    print(f"    - Train Samples      : {len(train_dataset)} (Expected: 21,193)")
    print(f"    - Validation Samples : {len(val_dataset)} (Expected: 2,649)")
    print(f"    - Test Samples       : {len(test_dataset)} (Expected: 2,650)")
    assert len(train_dataset) == 21193, f"Train split mismatch: {len(train_dataset)}"
    assert len(val_dataset) == 2649, f"Val split mismatch: {len(val_dataset)}"
    assert len(test_dataset) == 2650, f"Test split mismatch: {len(test_dataset)}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = create_collate_fn(tokenizer, max_length=MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Model Setup
    print(f"\n[*] Initializing DistilBERT from '{MODEL_NAME}' for 4-class classification...")
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)

    # 5. Optimizer & Scheduler
    total_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"[*] Hyperparameter Summary:")
    print(f"    - Epochs               : {NUM_EPOCHS}")
    print(f"    - Batch Size           : {BATCH_SIZE}")
    print(f"    - Learning Rate        : {LEARNING_RATE}")
    print(f"    - Weight Decay         : {WEIGHT_DECAY}")
    print(f"    - Total Steps          : {total_training_steps}")
    print(f"    - Warmup Steps         : {warmup_steps} ({WARMUP_RATIO*100:.1f}%)")
    print(f"    - Batches / Epoch      : {len(train_loader)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_duration_sec": []
    }

    best_val_accuracy = -1.0
    best_epoch = -1
    start_total_time = time.time()

    print("\n" + "=" * 70)
    print(" STARTING TRAINING LOOP (4 Epochs)")
    print("=" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                cur_loss = total_train_loss / step
                elapsed = time.time() - start_total_time
                print(f"  Epoch {epoch}/{NUM_EPOCHS} | Step {step:3d}/{len(train_loader)} | Running Loss: {cur_loss:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}s")

                # Update live progress file
                progress_data = {
                    "phase": "6.7",
                    "status": "TRAINING",
                    "current_epoch": epoch,
                    "total_epochs": NUM_EPOCHS,
                    "current_step": step,
                    "steps_per_epoch": len(train_loader),
                    "total_steps": total_training_steps,
                    "running_loss": round(cur_loss, 4),
                    "learning_rate": cur_lr,
                    "elapsed_seconds": round(elapsed, 1),
                    "best_epoch": best_epoch,
                    "best_val_accuracy": round(best_val_accuracy, 4) if best_val_accuracy > 0 else None
                }
                with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
                    json.dump(progress_data, pf, indent=2)

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        epoch_duration = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["epoch_duration_sec"].append(round(epoch_duration, 2))

        print(f"\n>>> Epoch {epoch:02d} Completed in {epoch_duration:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_acc * 100:.2f}%")

        # Save per-epoch checkpoint for recovery
        epoch_ckpt_dir = CHECKPOINT_DIR / f"epoch_{epoch}"
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            print(f"    [BEST] New highest validation accuracy! Saving best checkpoint to {CHECKPOINT_DIR}...")
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)
        print("-" * 70)

    total_training_time = time.time() - start_total_time
    print("\n" + "=" * 70)
    print(f" TRAINING COMPLETE in {total_training_time:.1f}s ({total_training_time / 60:.2f} min)")
    print(f" Best Checkpoint: Epoch {best_epoch} with Val Accuracy: {best_val_accuracy * 100:.2f}%")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 7. Evaluate on Held-out V3 Test Set (2,650 samples)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" EVALUATION ON HELD-OUT V3 TEST SET (2,650 samples)")
    print("=" * 70)

    print(f"[*] Loading best checkpoint from {CHECKPOINT_DIR}...")
    best_model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR).to(device)
    best_model.eval()

    test_preds, test_targets, test_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())
            test_probs.extend(probs)

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    acc = accuracy_score(test_targets, test_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2, 3])

    # Per-query latency benchmark (measured sequentially on 200 random test samples)
    print("[*] Profiling single-query inference latency...")
    test_texts = [item["text"] for item in test_dataset]
    sample_indices = random.sample(range(len(test_texts)), min(200, len(test_texts)))
    latencies_ms = []

    with torch.no_grad():
        for idx in sample_indices[:10]:
            enc = tokenizer(test_texts[idx], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
            _ = best_model(**enc)

        for idx in sample_indices:
            enc = tokenizer(test_texts[idx], return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
            t0 = time.perf_counter()
            _ = best_model(**enc)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    total_inf_time = sum(latencies_ms)

    per_class_metrics = {}
    for idx, cls_name in enumerate(TARGET_CLASSES):
        per_class_metrics[cls_name] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_sup[idx])
        }

    test_metrics_summary = {
        "model": "Fine-Tuned DistilBERT Phase 6.7 (Boundary Augmentation v3)",
        "dataset": "Antigravity Intent Classification Dataset (Phase 6.7 Boundary Augmentation v3)",
        "test_sample_count": len(test_dataset),
        "test_accuracy": round(float(acc), 4),
        "macro_avg": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4)
        },
        "weighted_avg": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1": round(float(weighted_f1), 4)
        },
        "per_class": per_class_metrics,
        "confusion_matrix": {
            "labels": TARGET_CLASSES,
            "matrix": cm.tolist()
        },
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "profiled_samples": len(sample_indices),
            "total_inference_time_ms": round(total_inf_time, 2)
        },
        "training_details": {
            "num_epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_val_accuracy": round(float(best_val_accuracy), 4),
            "total_training_time_sec": round(total_training_time, 2),
            "history": history,
            "checkpoint_path": str(CHECKPOINT_DIR),
            "device": str(device)
        }
    }

    test_metrics_path = REPORTS_DIR / "phase6_7_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics_summary, f, indent=2)
    print(f"[*] Saved V3 test metrics to {test_metrics_path}")

    # Plot Confusion Matrix
    cm_fig_path = FIGURES_DIR / "confusion_matrix_distilbert_phase6_7.png"
    plt.figure(figsize=(6.2, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES,
                linewidths=1.2, linecolor="#dddddd",
                annot_kws={"size": 12, "weight": "bold"})
    plt.xlabel("Predicted Intent Class", fontweight="bold", labelpad=8)
    plt.ylabel("Ground Truth Class", fontweight="bold", labelpad=8)
    plt.title(f"Phase 6.7 DistilBERT Test Confusion Matrix (N = {len(test_dataset)})", pad=14, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved confusion matrix figure to {cm_fig_path}")

    # -------------------------------------------------------------------------
    # 8. Evaluate on Dedicated Phase 6.7 Boundary Evaluation Set (200 cases)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" EVALUATING ON DEDICATED BOUNDARY EVALUATION SET (200 cases)")
    print("=" * 70)

    with open(HOLDOUT_BOUNDARY_FILE, "r", encoding="utf-8") as f:
        holdout_cases = json.load(f)

    b_targets, b_preds, b_probs, b_latencies = [], [], [], []
    b_case_details = []

    with torch.no_grad():
        for case in holdout_cases:
            c_id = case["case_id"]
            q_text = case["query"]
            expected = case["intent"]

            enc = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
            t0 = time.perf_counter()
            out = best_model(**enc)
            t1 = time.perf_counter()
            b_latencies.append((t1 - t0) * 1000.0)

            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(out.logits.argmax(dim=-1).cpu().item())
            pred_label = ID2LABEL[pred_id]
            conf = float(probs[pred_id])

            b_targets.append(LABEL2ID[expected])
            b_preds.append(pred_id)
            b_probs.append(probs)

            b_case_details.append({
                "case_id": c_id,
                "query": q_text,
                "expected": expected,
                "predicted": pred_label,
                "confidence": round(conf, 4),
                "is_correct": (pred_label == expected),
                "subtype": case.get("subtype", "unknown")
            })

    b_acc = accuracy_score(b_targets, b_preds)
    b_macro_p, b_macro_r, b_macro_f1, _ = precision_recall_fscore_support(b_targets, b_preds, average="macro", zero_division=0)
    b_per_class_p, b_per_class_r, b_per_class_f1, b_per_class_sup = precision_recall_fscore_support(b_targets, b_preds, labels=[2, 3], average=None, zero_division=0)
    b_cm = confusion_matrix(b_targets, b_preds, labels=[2, 3])

    sem_to_cstr_errors = sum(1 for d in b_case_details if d["expected"] == "SEMADPT" and d["predicted"] == "CSTRSET")
    cstr_to_sem_errors = sum(1 for d in b_case_details if d["expected"] == "CSTRSET" and d["predicted"] == "SEMADPT")

    print(f"[*] Dedicated Boundary Set Accuracy: {b_acc * 100:.2f}% ({sum(1 for d in b_case_details if d['is_correct'])}/200)")
    print(f"    - SEMADPT Precision: {b_per_class_p[0]*100:.2f}% | Recall: {b_per_class_r[0]*100:.2f}% | F1: {b_per_class_f1[0]*100:.2f}%")
    print(f"    - CSTRSET Precision: {b_per_class_p[1]*100:.2f}% | Recall: {b_per_class_r[1]*100:.2f}% | F1: {b_per_class_f1[1]*100:.2f}%")
    print(f"    - SEMADPT -> CSTRSET Errors: {sem_to_cstr_errors}")
    print(f"    - CSTRSET -> SEMADPT Errors: {cstr_to_sem_errors}")

    boundary_eval_summary = {
        "evaluation_set": "Phase 6.7 Dedicated Boundary Evaluation Set",
        "file": str(HOLDOUT_BOUNDARY_FILE),
        "total_cases": len(holdout_cases),
        "overall_accuracy": round(float(b_acc), 4),
        "macro_f1": round(float(b_macro_f1), 4),
        "per_class": {
            "SEMADPT": {
                "precision": round(float(b_per_class_p[0]), 4),
                "recall": round(float(b_per_class_r[0]), 4),
                "f1": round(float(b_per_class_f1[0]), 4),
                "support": int(b_per_class_sup[0])
            },
            "CSTRSET": {
                "precision": round(float(b_per_class_p[1]), 4),
                "recall": round(float(b_per_class_r[1]), 4),
                "f1": round(float(b_per_class_f1[1]), 4),
                "support": int(b_per_class_sup[1])
            }
        },
        "boundary_confusion": {
            "labels": ["SEMADPT", "CSTRSET"],
            "matrix": b_cm.tolist(),
            "semadpt_to_cstrset_errors": sem_to_cstr_errors,
            "cstrset_to_semadpt_errors": cstr_to_sem_errors
        },
        "latency": {
            "mean_ms": round(float(np.mean(b_latencies)), 2),
            "p50_ms": round(float(np.percentile(b_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(b_latencies, 95)), 2)
        },
        "case_details": b_case_details
    }

    boundary_metrics_path = REPORTS_DIR / "phase6_7_boundary_eval_metrics.json"
    with open(boundary_metrics_path, "w", encoding="utf-8") as f:
        json.dump(boundary_eval_summary, f, indent=2)
    print(f"[*] Saved boundary eval metrics to {boundary_metrics_path}")

    # Plot Boundary Confusion Matrix
    b_cm_fig_path = FIGURES_DIR / "phase6_7_boundary_confusion_matrix.png"
    plt.figure(figsize=(5.2, 4.5))
    sns.heatmap(b_cm, annot=True, fmt="d", cmap="Purples", cbar=True,
                xticklabels=["SEMADPT", "CSTRSET"], yticklabels=["SEMADPT", "CSTRSET"],
                linewidths=1.2, linecolor="#dddddd",
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel("Predicted Intent Class", fontweight="bold", labelpad=8)
    plt.ylabel("Ground Truth Class", fontweight="bold", labelpad=8)
    plt.title(f"Phase 6.7 Boundary Evaluation Confusion Matrix (N = 200)", pad=14, fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(b_cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[*] Saved boundary confusion matrix figure to {b_cm_fig_path}")

    # -------------------------------------------------------------------------
    # 9. Evaluate on Untouched 80-Query Hallucination Benchmark
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" EVALUATING CLASSIFIER ON 80-QUERY HALLUCINATION BENCHMARK (Classifier-Only)")
    print("=" * 70)

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        bench_queries = json.load(f)

    bench_results = []
    correct_count = 0
    class_correct = Counter()
    class_total = Counter()
    misclassified_queries = []

    with torch.no_grad():
        for q in bench_queries:
            q_id = q["query_id"]
            text = q["query"]
            expected = q["intent"]

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(device)
            out = best_model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(out.logits.argmax(dim=-1).cpu().item())
            pred_label = ID2LABEL[pred_id]
            conf = float(probs[pred_id])

            is_correct = (pred_label == expected)
            if is_correct:
                correct_count += 1
                class_correct[expected] += 1
            else:
                misclassified_queries.append({
                    "query_id": q_id,
                    "query": text,
                    "expected_label": expected,
                    "predicted_label": pred_label,
                    "confidence": round(conf, 4),
                    "class_probabilities": {ID2LABEL[i]: round(float(probs[i]), 4) for i in range(4)}
                })
            class_total[expected] += 1

            bench_results.append({
                "query_id": q_id,
                "query": text,
                "expected_label": expected,
                "predicted_label": pred_label,
                "confidence": round(conf, 4),
                "is_correct": is_correct
            })

    bench_acc = correct_count / len(bench_queries)
    print(f"\n[*] 80-Query Benchmark Classification Accuracy: {correct_count}/{len(bench_queries)} ({bench_acc*100:.2f}%)")
    for cls in TARGET_CLASSES:
        c = class_correct[cls]
        tot = class_total[cls]
        print(f"    - {cls:<10}: {c}/{tot} ({c/tot*100:.1f}%)")

    # Explicit audit of the five Phase 6.6 failure cases
    five_failures = ["SEM_03", "SEM_08", "SEM_12", "SEM_16", "SEM_17"]
    print("\n[*] Evaluation of Five Phase 6.6 Routed Failures (SEMADPT -> CSTRSET):")
    five_analysis = {}
    for fid in five_failures:
        res = next((x for x in bench_results if x["query_id"] == fid), None)
        if res:
            fixed = (res["predicted_label"] == "SEMADPT")
            five_analysis[fid] = {
                "query_id": fid,
                "query": res["query"],
                "expected": "SEMADPT",
                "predicted": res["predicted_label"],
                "confidence": res["confidence"],
                "is_fixed": fixed
            }
            status_str = "FIXED (SEMADPT)" if fixed else f"STILL FAILING ({res['predicted_label']})"
            print(f"    - {fid}: Predicted={res['predicted_label']} (conf={res['confidence']:.4f}) | Status: {status_str}")

    # Explicit audit of TXT_20 and SEM_20
    txt_20_res = next((x for x in bench_results if x["query_id"] == "TXT_20"), None)
    sem_20_res = next((x for x in bench_results if x["query_id"] == "SEM_20"), None)
    print("\n[*] Audit of Previously Fixed Queries (Phase 6.5 -> 6.7 Non-Regression):")
    txt_20_ok = (txt_20_res["predicted_label"] == "TXTSRC") if txt_20_res else False
    sem_20_ok = (sem_20_res["predicted_label"] == "SEMADPT") if sem_20_res else False
    print(f"    - TXT_20: Predicted={txt_20_res['predicted_label']} (conf={txt_20_res['confidence']:.4f}) | Maintained: {'YES' if txt_20_ok else 'REGRESSED'}")
    print(f"    - SEM_20: Predicted={sem_20_res['predicted_label']} (conf={sem_20_res['confidence']:.4f}) | Maintained: {'YES' if sem_20_ok else 'REGRESSED'}")

    benchmark_summary = {
        "benchmark_name": "Antigravity Hallucination Evaluation Benchmark (80 Queries)",
        "model": "Fine-Tuned DistilBERT Phase 6.7 (Boundary Augmentation v3)",
        "total_queries": len(bench_queries),
        "correct_count": correct_count,
        "accuracy": round(bench_acc, 4),
        "per_class_accuracy": {
            cls: {
                "correct": class_correct[cls],
                "total": class_total[cls],
                "accuracy": round(class_correct[cls] / class_total[cls], 4)
            } for cls in TARGET_CLASSES
        },
        "five_phase6_6_failures": five_analysis,
        "historical_fixed_queries": {
            "TXT_20": {
                "predicted": txt_20_res["predicted_label"] if txt_20_res else "",
                "confidence": txt_20_res["confidence"] if txt_20_res else 0.0,
                "is_maintained": txt_20_ok
            },
            "SEM_20": {
                "predicted": sem_20_res["predicted_label"] if sem_20_res else "",
                "confidence": sem_20_res["confidence"] if sem_20_res else 0.0,
                "is_maintained": sem_20_ok
            }
        },
        "all_misclassifications": misclassified_queries,
        "historical_comparison": {
            "phase5_1_benchmark_accuracy": 0.9250, # 74/80
            "phase6_5_benchmark_accuracy": 0.9375, # 75/80
            "phase6_7_benchmark_accuracy": round(bench_acc, 4)
        }
    }

    benchmark_metrics_path = REPORTS_DIR / "phase6_7_benchmark_classification.json"
    with open(benchmark_metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"[*] Saved benchmark classification metrics to {benchmark_metrics_path}")

    # Generate Markdown Benchmark Classification Report
    bench_md_path = REPORTS_DIR / "phase6_7_benchmark_classification.md"
    bench_md = f"""# Phase 6.7: 80-Query Benchmark Classification Report

**Evaluation Mode**: Classifier-Only (No Gemini API calls, No handlers)  
**Model**: `distilbert-base-uncased` fine-tuned on Dataset V3 (Boundary-Augmented)  
**Checkpoint**: [`artifacts/checkpoints/best_distilbert_phase6_7_boundary/`](file:///c:/Users/utham/Desktop/nndl%20proj/artifacts/checkpoints/best_distilbert_phase6_7_boundary/)  
**Overall Accuracy**: **{correct_count} / 80 ({bench_acc * 100:.2f}%)**  

---

## 1. Per-Class Accuracy

| Class | Correct / Total | Accuracy (%) | Historical Phase 6.5 | Delta vs Phase 6.5 |
| :--- | :---: | :---: | :---: | :---: |
| **`MATHDS`** | {class_correct['MATHDS']} / {class_total['MATHDS']} | {class_correct['MATHDS']/class_total['MATHDS']*100:.2f}% | 20 / 20 (100.0%) | {(class_correct['MATHDS']/class_total['MATHDS'] - 1.0)*100:+.2f}% |
| **`TXTSRC`** | {class_correct['TXTSRC']} / {class_total['TXTSRC']} | {class_correct['TXTSRC']/class_total['TXTSRC']*100:.2f}% | 20 / 20 (100.0%) | {(class_correct['TXTSRC']/class_total['TXTSRC'] - 1.0)*100:+.2f}% |
| **`SEMADPT`** | {class_correct['SEMADPT']} / {class_total['SEMADPT']} | {class_correct['SEMADPT']/class_total['SEMADPT']*100:.2f}% | 15 / 20 (75.0%) | {(class_correct['SEMADPT']/class_total['SEMADPT'] - 0.75)*100:+.2f}% |
| **`CSTRSET`** | {class_correct['CSTRSET']} / {class_total['CSTRSET']} | {class_correct['CSTRSET']/class_total['CSTRSET']*100:.2f}% | 20 / 20 (100.0%) | {(class_correct['CSTRSET']/class_total['CSTRSET'] - 1.0)*100:+.2f}% |
| **Total** | **{correct_count} / 80** | **{bench_acc*100:.2f}%** | **75 / 80 (93.75%)** | **{(bench_acc - 0.9375)*100:+.2f}%** |

---

## 2. Evaluation of the Five Phase 6.6 Failure Queries

| Query ID | Expected | Phase 6.5 Prediction | Phase 6.7 Prediction | Confidence | Fixed? | Query Text |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for fid in five_failures:
        fa = five_analysis[fid]
        bench_md += f"| **`{fid}`** | `{fa['expected']}` | `CSTRSET` | `{fa['predicted']}` | {fa['confidence']*100:.2f}% | {'**YES (FIXED)**' if fa['is_fixed'] else 'NO'} | {fa['query']} |\n"

    bench_md += f"""
---

## 3. Maintenance of Previous Fixes (Non-Regression)

| Query ID | Expected | Phase 5.1 Error | Phase 6.5 Status | Phase 6.7 Prediction | Confidence | Non-Regression Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TXT_20`** | `TXTSRC` | `MATHDS` (conf=0.5166) | Fixed (`TXTSRC`) | `{txt_20_res['predicted_label'] if txt_20_res else ''}` | {txt_20_res['confidence']*100:.2f}% | {'**MAINTAINED**' if txt_20_ok else '**REGRESSED**'} |
| **`SEM_20`** | `SEMADPT` | `CSTRSET` (conf=0.8047) | Fixed (`SEMADPT`) | `{sem_20_res['predicted_label'] if sem_20_res else ''}` | {sem_20_res['confidence']*100:.2f}% | {'**MAINTAINED**' if sem_20_ok else '**REGRESSED**'} |

---

## 4. Remaining Misclassifications (N = {len(misclassified_queries)})
"""
    if misclassified_queries:
        bench_md += "\n| Query ID | Ground Truth | Predicted | Confidence | Query Snippet |\n| :--- | :---: | :---: | :---: | :--- |\n"
        for m in misclassified_queries:
            bench_md += f"| `{m['query_id']}` | `{m['expected_label']}` | `{m['predicted_label']}` | {m['confidence']*100:.2f}% | {m['query'][:80]}... |\n"
    else:
        bench_md += "\n**Zero misclassifications on the 80-query benchmark (100.00% accuracy).**\n"

    with open(bench_md_path, "w", encoding="utf-8") as f:
        f.write(bench_md)
    print(f"[*] Saved benchmark classification markdown to {bench_md_path}")

    # -------------------------------------------------------------------------
    # 10. Comprehensive Phase 6.7 Experiment Report
    # -------------------------------------------------------------------------
    exp_report_path = REPORTS_DIR / "phase6_7_boundary_experiment_report.md"
    exp_report_md = f"""# Phase 6.7: SEMADPT/CSTRSET Boundary-Fix Experiment Report

**Project Title**: User Intent Classification for Hallucination Reduction in LLMs: A Cost-Accuracy Tradeoff Study  
**Phase**: 6.7 — Targeted SEMADPT/CSTRSET Boundary-Fix Experiment  
**Date**: October 2026  
**Status**: Executed & Verified  
**Model Architecture**: `distilbert-base-uncased` (Sequence Classification Head, 4 Classes)  
**Checkpoint Path**: [`artifacts/checkpoints/best_distilbert_phase6_7_boundary/`](file:///c:/Users/utham/Desktop/nndl%20proj/artifacts/checkpoints/best_distilbert_phase6_7_boundary/)  
**Dataset**: Phase 6.7 Boundary-Augmented Dataset V3 (26,492 Total Records)  
**Dedicated Holdout Evaluation Set**: [`data/hallucination_eval/phase6_7_boundary_eval.json`](file:///c:/Users/utham/Desktop/nndl%20proj/data/hallucination_eval/phase6_7_boundary_eval.json)  

---

## 1. Executive Summary & Experimental Objectives

In Phase 6.6, an end-to-end evaluation revealed that five of the six intent-routing failures (`SEM_03`, `SEM_08`, `SEM_12`, `SEM_16`, `SEM_17`) resulted from a systematic boundary misclassification:
- The classifier had learned that explicit structural/length constraints (e.g., exact sentence bounds, line counts, rhyming schemes, dialogue turn limits) strongly implied `CSTRSET`, even when the core user intent was creative or stylistic text generation (`SEMADPT`).
- Phase 6.7 executed a targeted boundary intervention by augmenting the training distribution with 800 paired boundary examples (400 `SEMADPT`, 400 `CSTRSET`) under source tag `BoundaryAugmentation_Phase6_7`, teaching the model that natural language transformation remains `SEMADPT` even with length constraints, while structured schema/field compliance defines `CSTRSET`.

---

## 2. Experimental Performance Across Three Distinct Evaluation Suites

### Suite 1: Standard Held-out Test Set (V2 vs. V3)
*Evaluates general taxonomy discrimination and detects regressions on non-boundary distributions.*

| Metric | Phase 6.5 Baseline (V2 Test, N=2,570) | Phase 6.7 Model (V3 Test, N=2,650) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.73% | **{acc*100:.2f}%** | **{(acc - 0.9973)*100:+.2f}%** |
| **Macro F1** | 99.72% | **{macro_f1*100:.2f}%** | **{(macro_f1 - 0.9972)*100:+.2f}%** |
| **Weighted F1** | 99.73% | **{weighted_f1*100:.2f}%** | **{(weighted_f1 - 0.9973)*100:+.2f}%** |
| **`MATHDS` F1** | 99.88% | **{per_class_metrics['MATHDS']['f1']*100:.2f}%** | {(per_class_metrics['MATHDS']['f1'] - 0.9988)*100:+.2f}% |
| **`TXTSRC` F1** | 99.54% | **{per_class_metrics['TXTSRC']['f1']*100:.2f}%** | {(per_class_metrics['TXTSRC']['f1'] - 0.9954)*100:+.2f}% |
| **`SEMADPT` F1** | 99.57% | **{per_class_metrics['SEMADPT']['f1']*100:.2f}%** | {(per_class_metrics['SEMADPT']['f1'] - 0.9957)*100:+.2f}% |
| **`CSTRSET` F1** | 99.90% | **{per_class_metrics['CSTRSET']['f1']*100:.2f}%** | {(per_class_metrics['CSTRSET']['f1'] - 0.9990)*100:+.2f}% |
| **Mean Latency** | 49.02 ms | **{mean_lat:.2f} ms** | {mean_lat - 49.02:+.2f} ms |
| **P50 Latency** | 38.34 ms | **{p50_lat:.2f} ms** | {p50_lat - 38.34:+.2f} ms |
| **P95 Latency** | 109.87 ms | **{p95_lat:.2f} ms** | {p95_lat - 109.87:+.2f} ms |

---

### Suite 2: Dedicated Boundary Holdout Evaluation Set (N = 200)
*Independent, unseen boundary holdout set (100 SEMADPT, 100 CSTRSET) with mixed constraint vocabulary.*

| Metric | Phase 6.7 Value |
| :--- | :---: |
| **Overall Boundary Accuracy** | **{b_acc*100:.2f}%** ({sum(1 for d in b_case_details if d['is_correct'])} / 200) |
| **`SEMADPT` Precision / Recall / F1** | {b_per_class_p[0]*100:.2f}% / {b_per_class_r[0]*100:.2f}% / **{b_per_class_f1[0]*100:.2f}%** |
| **`CSTRSET` Precision / Recall / F1** | {b_per_class_p[1]*100:.2f}% / {b_per_class_r[1]*100:.2f}% / **{b_per_class_f1[1]*100:.2f}%** |
| **`SEMADPT` $\\rightarrow$ `CSTRSET` Confusion Errors** | **{sem_to_cstr_errors}** |
| **`CSTRSET` $\\rightarrow$ `SEMADPT` Confusion Errors** | **{cstr_to_sem_errors}** |
| **Mean / P50 / P95 Latency** | {np.mean(b_latencies):.2f} ms / {np.percentile(b_latencies, 50):.2f} ms / {np.percentile(b_latencies, 95):.2f} ms |

---

### Suite 3: Untouched 80-Query Hallucination Benchmark (N = 80)
*Classifier-only evaluation on the historical benchmark.*

| Intent Class | Phase 6.5 Accuracy | Phase 6.7 Accuracy | Delta |
| :--- | :---: | :---: | :---: |
| **`MATHDS`** | 20 / 20 (100.0%) | **{class_correct['MATHDS']} / 20 ({class_correct['MATHDS']/class_total['MATHDS']*100:.1f}%)** | {(class_correct['MATHDS']/class_total['MATHDS'] - 1.0)*100:+.1f}% |
| **`TXTSRC`** | 20 / 20 (100.0%) | **{class_correct['TXTSRC']} / 20 ({class_correct['TXTSRC']/class_total['TXTSRC']*100:.1f}%)** | {(class_correct['TXTSRC']/class_total['TXTSRC'] - 1.0)*100:+.1f}% |
| **`SEMADPT`** | 15 / 20 (75.0%) | **{class_correct['SEMADPT']} / 20 ({class_correct['SEMADPT']/class_total['SEMADPT']*100:.1f}%)** | **{(class_correct['SEMADPT']/class_total['SEMADPT'] - 0.75)*100:+.1f}%** |
| **`CSTRSET`** | 20 / 20 (100.0%) | **{class_correct['CSTRSET']} / 20 ({class_correct['CSTRSET']/class_total['CSTRSET']*100:.1f}%)** | {(class_correct['CSTRSET']/class_total['CSTRSET'] - 1.0)*100:+.1f}% |
| **Total Benchmark** | **75 / 80 (93.75%)** | **{correct_count} / 80 ({bench_acc*100:.2f}%)** | **{(bench_acc - 0.9375)*100:+.2f}%** |

---

## 3. Targeted Audit: The Five Phase 6.6 Failure Queries

| Query ID | Expected | Phase 6.5 Prediction | Phase 6.7 Prediction | Confidence | Outcome |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`SEM_03`** | `SEMADPT` | `CSTRSET` (97.9%) | `{five_analysis['SEM_03']['predicted']}` | {five_analysis['SEM_03']['confidence']*100:.2f}% | {'**FIXED**' if five_analysis['SEM_03']['is_fixed'] else 'Unfixed'} |
| **`SEM_08`** | `SEMADPT` | `CSTRSET` (96.5%) | `{five_analysis['SEM_08']['predicted']}` | {five_analysis['SEM_08']['confidence']*100:.2f}% | {'**FIXED**' if five_analysis['SEM_08']['is_fixed'] else 'Unfixed'} |
| **`SEM_12`** | `SEMADPT` | `CSTRSET` (98.2%) | `{five_analysis['SEM_12']['predicted']}` | {five_analysis['SEM_12']['confidence']*100:.2f}% | {'**FIXED**' if five_analysis['SEM_12']['is_fixed'] else 'Unfixed'} |
| **`SEM_16`** | `SEMADPT` | `CSTRSET` (88.4%) | `{five_analysis['SEM_16']['predicted']}` | {five_analysis['SEM_16']['confidence']*100:.2f}% | {'**FIXED**' if five_analysis['SEM_16']['is_fixed'] else 'Unfixed'} |
| **`SEM_17`** | `SEMADPT` | `CSTRSET` (98.0%) | `{five_analysis['SEM_17']['predicted']}` | {five_analysis['SEM_17']['confidence']*100:.2f}% | {'**FIXED**' if five_analysis['SEM_17']['is_fixed'] else 'Unfixed'} |

### Non-Regression Audit:
- **`TXT_20`**: Ground Truth `TXTSRC` $\\rightarrow$ Predicted `{txt_20_res['predicted_label'] if txt_20_res else ''}` ({txt_20_res['confidence']*100:.2f}% confidence) | **{'MAINTAINED' if txt_20_ok else 'REGRESSED'}**
- **`SEM_20`**: Ground Truth `SEMADPT` $\\rightarrow$ Predicted `{sem_20_res['predicted_label'] if sem_20_res else ''}` ({sem_20_res['confidence']*100:.2f}% confidence) | **{'MAINTAINED' if sem_20_ok else 'REGRESSED'}**

---

## 4. Benchmark Integrity & Artifact Audit
- `eval_queries.json` SHA-256: `{EXPECTED_BENCHMARK_HASH}` (Verified Unchanged)
- `reference_corpus.json` SHA-256: `{EXPECTED_CORPUS_HASH}` (Verified Unchanged)
- `phase6_7_boundary_eval.json` SHA-256: `{EXPECTED_HOLDOUT_HASH}` (Verified Unchanged)
- Gemini API Calls: **0 (Strictly Classifier-Only Evaluation)**
"""

    with open(exp_report_path, "w", encoding="utf-8") as f:
        f.write(exp_report_md)
    print(f"[*] Saved experiment report to {exp_report_path}")

    # Post-evaluation Integrity Verification
    print("\n" + "=" * 70)
    print(" POST-EVALUATION BENCHMARK INTEGRITY VERIFICATION")
    print("=" * 70)
    verify_file_hash(BENCHMARK_FILE, EXPECTED_BENCHMARK_HASH, "eval_queries.json")
    verify_file_hash(CORPUS_FILE, EXPECTED_CORPUS_HASH, "reference_corpus.json")
    verify_file_hash(HOLDOUT_BOUNDARY_FILE, EXPECTED_HOLDOUT_HASH, "phase6_7_boundary_eval.json")
    print("=" * 70)
    print("PHASE 6.7 TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    train_phase6_7()
