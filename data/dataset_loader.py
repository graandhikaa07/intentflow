"""
Dataset Loader, Builder, and Stratified Splitter for Intent Classification.

Responsible for:
- Integrating CLINC-150 benchmark queries and curated domain-specific examples.
- Mapping all examples strictly to the 4 project classes:
  1. MATHDS  (Math, Algebra, Calculus, Statistics, Data Science)
  2. TXTSRC  (Factual Search, Knowledge Retrieval, Entity Lookups)
  3. SEMADPT (Semantic Adaptation, Creative Writing, Summarization, Tone Rewriting)
  4. CSTRSET (Constraint Satisfaction, Schemas, Length Bounds, Negative Constraints)
- Performing deterministic, stratified splitting:
  - 80% Train
  - 10% Validation
  - 10% Test
- Strictly verifying zero train/validation/test leakage and balanced class representation.
- Exporting splits to JSONL format:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl
- Providing PyTorch Dataset / DataLoader utilities for model training and evaluation.
"""

import os
import json
import random
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from data.generate_custom_data import (
    generate_mathds_samples,
    generate_txtsrc_samples,
    generate_semadpt_samples,
    generate_cstrset_samples
)

class IntentDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized intent classification records."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Any, max_seq_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class IntentDatasetLoader:
    def __init__(self, config_path: str = "configs/config.yaml", label_map_path: str = "configs/label_map.json"):
        self.config_path = config_path
        self.label_map_path = label_map_path
        self.config = self._load_config()
        self.label_map = self._load_label_map()
        self.seed = self.config.get("seed", 42)
        
        # Target classes
        self.classes = ["MATHDS", "TXTSRC", "SEMADPT", "CSTRSET"]
        self.raw_dir = self.config.get("dataset", {}).get("raw_data_dir", "data/raw")
        self.processed_dir = self.config.get("dataset", {}).get("processed_data_dir", "data/processed")
        
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_label_map(self) -> Dict[str, Any]:
        with open(self.label_map_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_clinc_data(self) -> Dict[str, List[str]]:
        """
        Loads CLINC-150 data and maps specific intents to target project classes:
        - 'calculator' -> MATHDS
        - Factual lookup intents -> TXTSRC
        - Conversational / style intents -> SEMADPT
        """
        clinc_cache = os.path.join(self.raw_dir, "clinc_full.json")
        if not os.path.exists(clinc_cache):
            raise FileNotFoundError(f"CLINC-150 file not found at {clinc_cache}")

        with open(clinc_cache, "r", encoding="utf-8") as f:
            data = json.load(f)

        clinc_mapped = {cls_name: [] for cls_name in self.classes}

        # Intent mapping definitions
        factual_intents = {
            "definition", "timezone", "calories", "distance",
            "weather", "date", "directions", "spelling",
            "gas_type", "international_fees"
        }
        conversational_intents = {
            "greeting", "goodbye", "thank_you", "tell_joke",
            "who_made_you", "what_are_your_hobbies"
        }

        # Extract items across all standard and OOS partitions in CLINC
        for split_key in ["train", "val", "test", "oos_train", "oos_val", "oos_test"]:
            for item in data.get(split_key, []):
                query_text, intent_name = item[0].strip(), item[1]
                if not query_text:
                    continue

                if intent_name == "calculator":
                    clinc_mapped["MATHDS"].append(query_text)
                elif intent_name in factual_intents:
                    clinc_mapped["TXTSRC"].append(query_text)
                elif intent_name in conversational_intents:
                    clinc_mapped["SEMADPT"].append(query_text)

        return clinc_mapped

    def build_balanced_dataset(self, target_per_class: int = 600) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
        """
        Builds clean, balanced, and globally deduplicated pools for each of the 4 classes.
        Combines CLINC-150 queries with domain-specific curated generators.
        
        Ensures SEMADPT has a balanced composition of:
        - Creative generation (~140 samples)
        - Tone transformation (~140 samples)
        - Summarization (~140 samples)
        - Open-ended conversational adaptation from CLINC-150 (~180 samples)
        
        Ensures CSTRSET has pure structural and schema constraints without creative writing overlap.
        Pre-reserves representative boundary cases into test splits.
        """
        rng = random.Random(self.seed)
        clinc_data = self._load_clinc_data()

        # Generate custom programmatic samples
        custom_data = {
            "MATHDS": generate_mathds_samples(seed=self.seed),
            "TXTSRC": generate_txtsrc_samples(seed=self.seed),
            "SEMADPT": generate_semadpt_samples(seed=self.seed),
            "CSTRSET": generate_cstrset_samples(seed=self.seed),
        }

        # Representative boundary cases mandated for the held-out test set
        representative_test_queries = {
            "SEMADPT": [
                "Rewrite this sentence in a friendly tone.",
                "Write a short creative story about a rainy evening.",
                "Summarize this paragraph.",
                "Make this message sound more professional.",
            ],
            "CSTRSET": [
                "Return the information as JSON with fields name, age, and city.",
                "Give exactly 3 items in a JSON array.",
                "Return valid YAML only.",
                "Do not include any additional text.",
            ],
            "MATHDS": [
                "What is 25 * 48?",
                "Solve x^2 - 5x + 6 = 0",
            ],
            "TXTSRC": [
                "What is the capital of Japan?",
                "What is the boiling point of water?",
            ]
        }

        global_seen_texts = set()
        test_reserved = {cls: [] for cls in self.classes}
        
        # Pre-register representative boundary test queries
        for cls_name, q_list in representative_test_queries.items():
            for q in q_list:
                clean_q = q.strip()
                global_seen_texts.add(clean_q.lower())
                test_reserved[cls_name].append({"text": clean_q, "label": cls_name})

        class_pools = {}

        for cls_name in self.classes:
            clinc_pool = clinc_data.get(cls_name, [])
            custom_pool = custom_data.get(cls_name, [])
            rng.shuffle(clinc_pool)
            rng.shuffle(custom_pool)

            selected = []
            needed_from_pool = target_per_class - len(test_reserved[cls_name])

            if cls_name == "SEMADPT":
                # Ensure balanced mixture: 180 from CLINC conversational intents, 416 from custom creative/tone/summarization
                clinc_target = 180
                custom_target = needed_from_pool - clinc_target

                # 1. Custom creative / tone / summarization samples
                for text in custom_pool:
                    clean_t = text.strip()
                    norm = clean_t.lower()
                    if norm not in global_seen_texts and len(clean_t) > 5:
                        global_seen_texts.add(norm)
                        selected.append({"text": clean_t, "label": cls_name})
                        if len(selected) >= custom_target:
                            break

                # 2. CLINC conversational chitchat samples
                clinc_count = 0
                for text in clinc_pool:
                    clean_t = text.strip()
                    norm = clean_t.lower()
                    if norm not in global_seen_texts and len(clean_t) > 5:
                        global_seen_texts.add(norm)
                        selected.append({"text": clean_t, "label": cls_name})
                        clinc_count += 1
                        if clinc_count >= clinc_target:
                            break

            elif cls_name == "CSTRSET":
                # Pure constraint/schema queries
                for text in custom_pool:
                    clean_t = text.strip()
                    norm = clean_t.lower()
                    if norm not in global_seen_texts and len(clean_t) > 5:
                        global_seen_texts.add(norm)
                        selected.append({"text": clean_t, "label": cls_name})
                        if len(selected) >= needed_from_pool:
                            break

            else:
                # MATHDS and TXTSRC: Combine CLINC and custom pools
                combined = clinc_pool + custom_pool
                for text in combined:
                    clean_t = text.strip()
                    norm = clean_t.lower()
                    if norm not in global_seen_texts and len(clean_t) > 5:
                        global_seen_texts.add(norm)
                        selected.append({"text": clean_t, "label": cls_name})
                        if len(selected) >= needed_from_pool:
                            break

            if len(selected) < needed_from_pool:
                raise ValueError(
                    f"Insufficient unique samples for class '{cls_name}'. "
                    f"Got {len(selected)}, required {needed_from_pool}."
                )

            class_pools[cls_name] = selected

        return class_pools, test_reserved

    def prepare_and_save_splits(self, target_per_class: int = 600) -> Dict[str, Any]:
        """
        Executes the Phase 5.1 dataset preparation and stratified splitting workflow:
        1. Compiles balanced 4-class pools (600/class, 2400 total).
        2. Includes representative boundary test cases in test split.
        3. Performs reproducible split: 80% train (1920), 10% val (240), 10% test (240).
        4. Saves train.jsonl, val.jsonl, test.jsonl.
        5. Verifies zero train/test leakage and balance.
        6. Returns and prints comprehensive dataset metrics report.
        """
        rng = random.Random(self.seed)
        class_pools, test_reserved = self.build_balanced_dataset(target_per_class=target_per_class)

        train_data = []
        val_data = []
        test_data = []

        for cls_name in self.classes:
            pool = class_pools[cls_name]
            rng.shuffle(pool)

            reserved = test_reserved[cls_name]
            needed_test = 60 - len(reserved)  # 10% of 600 = 60
            val_needed = 60                   # 10% of 600 = 60
            train_needed = 480                # 80% of 600 = 480

            cls_test = reserved + pool[:needed_test]
            cls_val = pool[needed_test:needed_test + val_needed]
            cls_train = pool[needed_test + val_needed:needed_test + val_needed + train_needed]

            assert len(cls_train) == 480, f"Expected 480 train for {cls_name}, got {len(cls_train)}"
            assert len(cls_val) == 60, f"Expected 60 val for {cls_name}, got {len(cls_val)}"
            assert len(cls_test) == 60, f"Expected 60 test for {cls_name}, got {len(cls_test)}"

            train_data.extend(cls_train)
            val_data.extend(cls_val)
            test_data.extend(cls_test)

        rng.shuffle(train_data)
        rng.shuffle(val_data)
        rng.shuffle(test_data)

        # Save to disk
        train_path = os.path.join(self.processed_dir, "train.jsonl")
        val_path = os.path.join(self.processed_dir, "val.jsonl")
        test_path = os.path.join(self.processed_dir, "test.jsonl")

        self._save_jsonl(train_data, train_path)
        self._save_jsonl(val_data, val_path)
        self._save_jsonl(test_data, test_path)

        # Verification & Leakage checks
        train_set = {d["text"].strip().lower() for d in train_data}
        val_set = {d["text"].strip().lower() for d in val_data}
        test_set = {d["text"].strip().lower() for d in test_data}

        train_val_overlap = len(train_set.intersection(val_set))
        train_test_overlap = len(train_set.intersection(test_set))
        val_test_overlap = len(val_set.intersection(test_set))
        total_overlap = train_val_overlap + train_test_overlap + val_test_overlap

        dataset = train_data + val_data + test_data
        labels = [d["label"] for d in dataset]
        train_labels = [d["label"] for d in train_data]
        val_labels = [d["label"] for d in val_data]
        test_labels = [d["label"] for d in test_data]

        # Build report statistics
        report = {
            "total_samples": len(dataset),
            "target_classes": self.classes,
            "overall_class_distribution": dict(Counter(labels)),
            "split_counts": {
                "train": len(train_data),
                "validation": len(val_data),
                "test": len(test_data)
            },
            "split_class_distribution": {
                "train": dict(Counter(train_labels)),
                "validation": dict(Counter(val_labels)),
                "test": dict(Counter(test_labels))
            },
            "duplicate_count_across_splits": total_overlap,
            "zero_leakage_confirmed": (total_overlap == 0),
            "file_paths": {
                "train": train_path,
                "val": val_path,
                "test": test_path
            }
        }

        self._print_report(report)
        return report

    def _save_jsonl(self, data: List[Dict[str, str]], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _print_report(self, report: Dict[str, Any]):
        print("\n" + "=" * 70)
        print(" PHASE 1: DATASET PREPARATION & STRATIFIED SPLIT REPORT")
        print("=" * 70)
        print(f"Total Clean Samples : {report['total_samples']}")
        print(f"Target Classes      : {', '.join(report['target_classes'])}")
        print("\nOverall Class Distribution:")
        for cls_name, count in report['overall_class_distribution'].items():
            pct = (count / report['total_samples']) * 100
            print(f"  - {cls_name:<10}: {count:>5} ({pct:.1f}%)")

        print("\nSplit Counts (80% Train / 10% Val / 10% Test):")
        for split_name, count in report['split_counts'].items():
            pct = (count / report['total_samples']) * 100
            print(f"  - {split_name.capitalize():<12}: {count:>5} ({pct:.1f}%)")

        print("\nSamples Per Class in Each Split:")
        header = f"  {'Class':<10} | {'Train (80%)':<12} | {'Val (10%)':<10} | {'Test (10%)':<10} | {'Total':<6}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cls_name in report['target_classes']:
            tr = report['split_class_distribution']['train'].get(cls_name, 0)
            va = report['split_class_distribution']['validation'].get(cls_name, 0)
            te = report['split_class_distribution']['test'].get(cls_name, 0)
            tot = tr + va + te
            print(f"  {cls_name:<10} | {tr:>12} | {va:>10} | {te:>10} | {tot:>6}")

        print("\nData Leakage & Integrity Verification:")
        print(f"  - Duplicate queries across splits : {report['duplicate_count_across_splits']}")
        print(f"  - Train/Val/Test Leakage Free     : {'CONFIRMED [YES]' if report['zero_leakage_confirmed'] else 'FAILED [NO]'}")
        print("\nSaved Output Files:")
        for split, path in report['file_paths'].items():
            print(f"  - {split.capitalize():<12}: {path}")
        print("=" * 70 + "\n")

    def load_processed_split(self, split: str = "train") -> Tuple[List[str], List[int]]:
        """Loads a processed JSONL split and maps label strings to integer IDs."""
        split_map = {
            "train": os.path.join(self.processed_dir, "train.jsonl"),
            "val": os.path.join(self.processed_dir, "val.jsonl"),
            "test": os.path.join(self.processed_dir, "test.jsonl"),
        }
        if split not in split_map:
            raise ValueError(f"Split '{split}' must be one of ['train', 'val', 'test']")

        file_path = split_map[split]
        texts, label_ids = [], []
        label2id = self.label_map["label2id"]

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                texts.append(record["text"])
                label_ids.append(label2id[record["label"]])

        return texts, label_ids

    def get_dataloaders(self, tokenizer: Any, batch_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        Constructs PyTorch DataLoaders for train, val, and test splits.
        Ready for seamless integration into Phase 2 DistilBERT training.
        """
        bs = batch_size or self.config.get("dataset", {}).get("batch_size", 32)
        max_len = self.config.get("dataset", {}).get("max_seq_length", 128)

        loaders = {}
        for split in ["train", "val", "test"]:
            texts, label_ids = self.load_processed_split(split)
            ds = IntentDataset(texts, label_ids, tokenizer, max_seq_length=max_len)
            shuffle = (split == "train")
            loaders[split] = DataLoader(ds, batch_size=bs, shuffle=shuffle)

        return loaders


if __name__ == "__main__":
    loader = IntentDatasetLoader()
    loader.prepare_and_save_splits(target_per_class=600)
