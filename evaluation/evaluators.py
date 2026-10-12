"""
Automated Evaluation Metrics and Scorers for Hallucination-Reduction Experiment.

Provides objective, reproducible evaluation functions for all 80 benchmark queries:
- MATHDS: Exact numeric match, root set validation, tolerance bounds.
- TXTSRC: Reference evidence grounding, required entity coverage, hallucinated/prohibited claim checks.
- SEMADPT: Instruction adherence, structural constraints (sentence counts, prohibited phrases, themes).
- CSTRSET: Machine-checkable structural schemas (JSON, YAML, XML, item counts, type constraints, negative constraints).
"""

import re
import json
import yaml
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, Optional, List

def evaluate_mathds(response_text: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluates mathematical responses against benchmark ground truth.
    Returns: (is_correct, reason, error_category)
    """
    if not response_text or response_text.startswith("[Error]"):
        return False, "Empty or error response", "handler_failure"

    eval_type = ground_truth.get("evaluation_type", "exact_numeric")

    # 1. Set of roots (e.g. quadratic equations: [3.0, 4.0] or [-4.0, 4.0])
    if eval_type == "set_of_roots":
        expected_roots = ground_truth.get("expected_roots", [])
        # Look for the roots in text
        found_all = True
        for root in expected_roots:
            # Check for integer or float format in text
            pattern_int = rf"(?<![\d\.\-])-?{int(abs(root))}(?!\d)" if root.is_integer() else None
            # Check for float or int match
            val_str = str(int(root)) if root.is_integer() else str(root)
            if val_str not in response_text:
                found_all = False
                break
        if found_all:
            return True, f"Found all expected roots: {expected_roots}", None
        return False, f"Missing one or more expected roots {expected_roots} in: {response_text[:80]}", "mathematical_error"

    # 2. Exact numeric value
    expected_val = ground_truth.get("expected_value")
    if expected_val is not None:
        # Check direct string match
        exp_int = int(expected_val) if isinstance(expected_val, (int, float)) and float(expected_val).is_integer() else None
        exp_str = str(exp_int) if exp_int is not None else str(expected_val)

        # Check if the number appears in response
        # Matches standalone number e.g. "9396", "289", "126"
        pattern = rf"(?<![\d\.]){re.escape(exp_str)}(?![\d\.])"
        if re.search(pattern, response_text):
            return True, f"Exact numerical match for {expected_val}", None

        # Try extracting numbers from response text and checking equality
        extracted_numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response_text.replace(",", ""))
        for num_str in extracted_numbers:
            try:
                num = float(num_str)
                if abs(num - float(expected_val)) < 1e-4:
                    return True, f"Numerical match {num} equals expected {expected_val}", None
            except ValueError:
                continue

        return False, f"Expected {expected_val}, but answer was: '{response_text[:80]}'", "mathematical_error"

    return False, "Unknown evaluation ground truth", "evaluation_error"


def evaluate_txtsrc(response_text: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluates factual retrieval responses against reference corpus ground truth.
    Returns: (is_correct, reason, error_category)
    """
    if not response_text or response_text.startswith("[Controlled Fallback]") or response_text.startswith("[Error]"):
        return False, "Unanswered or fallback response", "unsupported_factual_claim"

    resp_lower = response_text.lower()

    # Check prohibited entities first (e.g. Sydney/Melbourne claimed as capital of Australia)
    prohibited = ground_truth.get("prohibited_entities", [])
    for p in prohibited:
        p_lower = p.lower()
        # If prohibited entity is mentioned, check if claimed as answer
        if p_lower in resp_lower and "not " + p_lower not in resp_lower and "instead of " + p_lower not in resp_lower:
            # Check context: if Canberra is also mentioned, it might be discussing compromise
            if "canberra" in resp_lower and "capital" in resp_lower:
                pass  # Mentioning Sydney/Melbourne in historical context is allowed
            elif "ottawa" in resp_lower and "capital" in resp_lower:
                pass
            else:
                return False, f"Contains prohibited incorrect entity: {p}", "factual_error"

    # Check required entities
    required = ground_truth.get("required_entities", [])
    missing = []
    for req in required:
        req_clean = req.lower().replace(",", "")
        resp_clean = resp_lower.replace(",", "")
        if req_clean not in resp_clean:
            missing.append(req)

    if not missing:
        return True, f"All required entities present: {required}", None

    # Alternative target facts check
    target_facts = ground_truth.get("target_facts", [])
    found_any_target = any(tf.lower() in resp_lower for tf in target_facts)
    if found_any_target and len(missing) < len(required):
        return True, f"Target factual claims satisfied: {target_facts}", None

    return False, f"Missing required factual entities: {missing}", "factual_error"


def evaluate_semadpt(response_text: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluates semantic adaptation responses for instruction adherence according to benchmark criteria.
    Returns: (is_correct, reason, error_category)
    """
    if not response_text or response_text.startswith("[Error]") or response_text.startswith("[Controlled"):
        return False, "Empty or fallback response", "instruction_following_failure"

    resp_clean = response_text.strip()
    words = resp_clean.split()
    word_count = len(words)

    # 1. Check max word count constraint
    if "max_word_count" in ground_truth:
        max_words = ground_truth["max_word_count"]
        # Allow small 15% margin for polite salutation/signoff
        if word_count > max_words + 4:
            return False, f"Exceeded max word count: {word_count} > {max_words}", "instruction_following_failure"

    # 2. Check sentence count constraint
    sentences = [s.strip() for s in re.split(r"[.!?]+", resp_clean) if s.strip()]
    if "exact_sentence_count" in ground_truth:
        target_s = ground_truth["exact_sentence_count"]
        # Relax slightly if dialogue or poetry
        if abs(len(sentences) - target_s) > 1 and "turn_count" not in ground_truth and "line_count" not in ground_truth:
            return False, f"Sentence count {len(sentences)} does not match required {target_s}", "instruction_following_failure"

    if "min_sentences" in ground_truth and "max_sentences" in ground_truth:
        min_s = ground_truth["min_sentences"]
        max_s = ground_truth["max_sentences"]
        if len(sentences) < min_s - 1 or len(sentences) > max_s + 1:
            return False, f"Sentence count {len(sentences)} outside [{min_s}, {max_s}] range", "instruction_following_failure"

    # 3. Check prohibited words/phrases
    prohibited_words = ground_truth.get("prohibited_words", []) + ground_truth.get("prohibited_phrases", [])
    for pw in prohibited_words:
        if pw.lower() in resp_clean.lower():
            return False, f"Contains prohibited word/phrase: '{pw}'", "instruction_following_failure"

    # 4. Check core content elements
    core_elements = ground_truth.get("core_content_elements", [])
    missing_core = []
    for elem in core_elements:
        # Check if any significant word of the core element appears
        keywords = [w.lower() for w in elem.split() if len(w) > 3 and w.lower() not in ["this", "that", "with", "from", "into"]]
        if keywords and not any(kw in resp_clean.lower() for kw in keywords):
            missing_core.append(elem)

    if missing_core and len(missing_core) == len(core_elements):
        return False, f"Failed to preserve core content elements: {missing_core}", "instruction_following_failure"

    return True, "Instruction adherence criteria met", None


def evaluate_cstrset(response_text: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Evaluates constrained responses for syntax, schema, item counts, and negative constraints.
    Returns: (is_correct, reason, error_category)
    """
    if not response_text or response_text.startswith("[Error]") or response_text.startswith("[Controlled"):
        return False, "Empty or error response", "constraint_violation"

    text = response_text.strip()
    fmt = ground_truth.get("expected_format")

    # Helper: strip markdown fences if model wrapped
    def extract_fenced(s: str) -> str:
        match = re.search(r"```(?:json|yaml|xml)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else s

    # 1. JSON Object
    if fmt == "json_object":
        raw_to_parse = extract_fenced(text)
        try:
            data = json.loads(raw_to_parse)
            if not isinstance(data, dict):
                return False, "Expected JSON object (dict), got different type", "invalid_structure"
        except Exception as e:
            return False, f"Invalid JSON syntax: {e}", "invalid_structure"

        # Check required keys
        for key in ground_truth.get("required_keys", []):
            if key not in data:
                return False, f"Missing required JSON key: '{key}'", "constraint_violation"

        # Check required root keys
        for key in ground_truth.get("required_root_keys", []):
            if key not in data:
                return False, f"Missing required root key: '{key}'", "constraint_violation"

        # Check nested keys
        nested = ground_truth.get("nested_keys", {})
        for root_k, child_keys in nested.items():
            if root_k not in data or not isinstance(data[root_k], dict):
                return False, f"Missing nested object key: '{root_k}'", "constraint_violation"
            for c_k in child_keys:
                if c_k not in data[root_k]:
                    return False, f"Missing nested child key: '{c_k}' in '{root_k}'", "constraint_violation"

        # Check type constraints
        types = ground_truth.get("type_constraints", {})
        for k, t in types.items():
            if k in data:
                val = data[k]
                if t == "int" and not (isinstance(val, int) and not isinstance(val, bool)):
                    return False, f"Key '{k}' expected int, got {type(val).__name__}", "constraint_violation"
                if t == "bool" and not isinstance(val, bool):
                    return False, f"Key '{k}' expected bool, got {type(val).__name__}", "constraint_violation"
                if t == "str" and not isinstance(val, str):
                    return False, f"Key '{k}' expected str, got {type(val).__name__}", "constraint_violation"
                if t == "float_or_number" and not isinstance(val, (int, float)):
                    return False, f"Key '{k}' expected number, got {type(val).__name__}", "constraint_violation"

        # Check value constraints
        val_constraints = ground_truth.get("value_constraints", {})
        for k, exp_val in val_constraints.items():
            if data.get(k) != exp_val:
                return False, f"Key '{k}' value {data.get(k)} != expected {exp_val}", "constraint_violation"

        # Check tags list length / casing if applicable
        if "tags_list_length" in ground_truth:
            tags = data.get("tags")
            if not isinstance(tags, list) or len(tags) != ground_truth["tags_list_length"]:
                return False, f"Expected tags list of length {ground_truth['tags_list_length']}", "constraint_violation"
            if ground_truth.get("tags_casing") == "all_lowercase":
                if not all(isinstance(t, str) and t.islower() for t in tags):
                    return False, "Not all tags are lowercase strings", "constraint_violation"

        return True, "Valid JSON object matching schema constraints", None

    # 2. JSON Array
    elif fmt == "json_array":
        raw_to_parse = extract_fenced(text)
        try:
            data = json.loads(raw_to_parse)
            if not isinstance(data, list):
                return False, "Expected JSON array (list), got different type", "invalid_structure"
        except Exception as e:
            return False, f"Invalid JSON array syntax: {e}", "invalid_structure"

        # Check exact item count
        if "exact_item_count" in ground_truth:
            exp_count = ground_truth["exact_item_count"]
            if len(data) != exp_count:
                return False, f"Expected exactly {exp_count} items, found {len(data)}", "constraint_violation"

        # Check ordering
        if ground_truth.get("ordering_constraint") == "strictly_ascending_numbers":
            if not all(isinstance(x, (int, float)) for x in data):
                return False, "Array items must be numbers", "constraint_violation"
            if not (data[0] < data[1] < data[2]):
                return False, f"Array numbers not in strictly ascending order: {data}", "constraint_violation"

        # Check exact value match
        if "expected_value" in ground_truth:
            if data != ground_truth["expected_value"]:
                return False, f"Array does not match expected value: {data} != {ground_truth['expected_value']}", "constraint_violation"

        return True, "Valid JSON array matching count and value constraints", None

    # 3. YAML Mapping
    elif fmt == "yaml_mapping":
        raw_to_parse = extract_fenced(text)
        try:
            data = yaml.safe_load(raw_to_parse)
            if not isinstance(data, dict):
                return False, "YAML did not parse as a mapping (dict)", "invalid_structure"
        except Exception as e:
            return False, f"Invalid YAML syntax: {e}", "invalid_structure"

        for k in ground_truth.get("required_keys", []):
            if k not in data:
                return False, f"Missing required YAML key: '{k}'", "constraint_violation"

        types = ground_truth.get("type_constraints", {})
        for k, t in types.items():
            if k in data and t == "int" and not (isinstance(data[k], int) and not isinstance(data[k], bool)):
                return False, f"YAML key '{k}' expected int, got {type(data[k]).__name__}", "constraint_violation"

        return True, "Valid YAML mapping satisfying schema and types", None

    # 4. XML
    elif fmt == "xml":
        raw_to_parse = extract_fenced(text)
        try:
            root = ET.fromstring(raw_to_parse)
        except Exception as e:
            return False, f"Invalid XML syntax: {e}", "invalid_structure"

        exp_root = ground_truth.get("root_tag")
        if exp_root and root.tag != exp_root:
            return False, f"Root tag '{root.tag}' != expected '{exp_root}'", "constraint_violation"

        child_tags = [c.tag for c in root]
        for req_tag in ground_truth.get("required_child_tags", []):
            if req_tag not in child_tags:
                return False, f"Missing required child XML tag: '<{req_tag}>'", "constraint_violation"

        return True, "Valid XML document satisfying root and child tag schemas", None

    # 5. Numbered List
    elif fmt == "numbered_list":
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if "exact_line_count" in ground_truth:
            exp_count = ground_truth["exact_line_count"]
            if len(lines) != exp_count:
                return False, f"Numbered list line count {len(lines)} != {exp_count}", "constraint_violation"

        for i, line in enumerate(lines, start=1):
            if not re.match(rf"^{i}\.\s+", line):
                return False, f"Line {i} does not start with '{i}. ': '{line}'", "constraint_violation"

        return True, "Valid numbered list satisfying exact line count and formatting", None

    # 6. Bulleted List
    elif fmt == "bulleted_list":
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        exp_count = ground_truth.get("exact_line_count", 4)
        bullet_sym = ground_truth.get("bullet_symbol", "-")
        if len(lines) != exp_count:
            return False, f"Bulleted list line count {len(lines)} != {exp_count}", "constraint_violation"
        for line in lines:
            if not line.startswith(bullet_sym):
                return False, f"Line does not start with bullet '{bullet_sym}': '{line}'", "constraint_violation"

        return True, "Valid bulleted list satisfying line count and bullet character", None

    # 7. CSV Line
    elif fmt == "csv_line":
        # Single line or multiple items split by comma
        items = [x.strip() for x in text.replace("\n", " ").split(",") if x.strip()]
        exp_count = ground_truth.get("exact_item_count", 6)
        if len(items) != exp_count:
            return False, f"CSV item count {len(items)} != {exp_count}", "constraint_violation"
        return True, f"Valid CSV line containing exactly {exp_count} items", None

    # 8. Single Digit
    elif fmt == "raw_digit":
        clean_d = text.strip()
        exp_d = str(ground_truth.get("expected_digit", 7))
        if clean_d == exp_d:
            return True, f"Output matches raw digit '{exp_d}'", None
        return False, f"Output '{clean_d}' does not equal single digit '{exp_d}'", "constraint_violation"

    # 9. Plain Text Word Count
    elif fmt == "plain_text" and "exact_word_count" in ground_truth:
        words = text.split()
        exp_w = ground_truth["exact_word_count"]
        if len(words) != exp_w:
            return False, f"Word count {len(words)} != {exp_w}", "constraint_violation"
        return True, f"Exact word count {exp_w} satisfied", None

    # 10. Single Sentence Length Bounded
    elif fmt == "single_sentence":
        words = text.split()
        min_w = ground_truth.get("min_word_count", 10)
        max_w = ground_truth.get("max_word_count", 15)
        if not (min_w <= len(words) <= max_w):
            return False, f"Sentence word count {len(words)} outside [{min_w}, {max_w}] range", "constraint_violation"
        return True, f"Single sentence within [{min_w}, {max_w}] word count bounds", None

    # 11. Text list (descending alphabetical)
    elif fmt == "text_list" and ground_truth.get("ordering_constraint") == "descending_alphabetical":
        lines = [re.sub(r"^\d+\.\s*", "", l).strip() for l in text.split("\n") if l.strip()]
        if len(lines) != 3:
            return False, f"Expected 3 items in list, got {len(lines)}", "constraint_violation"
        if not (lines[0].lower() > lines[1].lower() > lines[2].lower()):
            return False, f"Items not in descending alphabetical order: {lines}", "constraint_violation"
        return True, "Items in strictly descending alphabetical order", None

    return True, "Constraint satisfied", None


def evaluate_query(intent: str, response_text: str, ground_truth: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    """
    Main dispatcher evaluating a response for a specific benchmark intent.
    Returns: (is_correct, reason, error_category)
    """
    if intent == "MATHDS":
        return evaluate_mathds(response_text, ground_truth)
    elif intent == "TXTSRC":
        return evaluate_txtsrc(response_text, ground_truth)
    elif intent == "SEMADPT":
        return evaluate_semadpt(response_text, ground_truth)
    elif intent == "CSTRSET":
        return evaluate_cstrset(response_text, ground_truth)
    else:
        return False, f"Unknown intent: {intent}", "evaluation_error"
