"""
Math Solver Handler (MATHDS).

Responsible for:
- Handling mathematical, arithmetic, and quantitative queries.
- Parsing mathematical expressions and running deterministic calculation via SymPy / Python.
- Eliminating LLM calculation hallucination by offloading arithmetic to deterministic execution.
"""

import re
import math
import statistics
import time
from typing import Dict, Any, List
import sympy as sp

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

class MathSolverHandler:
    def __init__(self):
        self.name = "math_solver"

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Extracts mathematical expressions or equations, executes them deterministically,
        and returns the exact calculation result without LLM generation.
        """
        start_time = time.perf_counter()
        q = query.strip()

        # 1. Statistics: mean, median, population variance
        # Arithmetic mean: "Find the arithmetic mean of the numbers 12, 18, 25, 31, and 44."
        mean_match = re.search(r"(?:arithmetic\s+mean|average)\s+(?:of\s+(?:the\s+numbers\s+)?)?([\d\s,]+(?:and\s+\d+)?)", q, re.IGNORECASE)
        if mean_match:
            nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", mean_match.group(1))]
            if nums:
                val = statistics.mean(nums)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "status": "success",
                    "method": "deterministic_statistics (mean)",
                    "parsed_expression": f"mean({nums})",
                    "result": val,
                    "formatted_answer": str(int(val) if val.is_integer() else val),
                    "execution_latency_ms": round(elapsed_ms, 2)
                }

        # Median: "What is the median of the dataset: 7, 3, 9, 12, 1, 15, 6?"
        median_match = re.search(r"median\s+(?:of\s+(?:the\s+dataset:?\s*)?)?([\d\s,]+(?:and\s+\d+)?)", q, re.IGNORECASE)
        if median_match:
            nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", median_match.group(1))]
            if nums:
                val = statistics.median(nums)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "status": "success",
                    "method": "deterministic_statistics (median)",
                    "parsed_expression": f"median({nums})",
                    "result": val,
                    "formatted_answer": str(int(val) if val.is_integer() else val),
                    "execution_latency_ms": round(elapsed_ms, 2)
                }

        # Population variance: "Calculate the population variance of the values 2, 4, 6, 8, and 10."
        pvar_match = re.search(r"population\s+variance\s+(?:of\s+(?:the\s+values:?\s*)?)?([\d\s,]+(?:and\s+\d+)?)", q, re.IGNORECASE)
        if pvar_match:
            nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", pvar_match.group(1))]
            if nums:
                val = statistics.pvariance(nums)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "status": "success",
                    "method": "deterministic_statistics (population_variance)",
                    "parsed_expression": f"pvariance({nums})",
                    "result": val,
                    "formatted_answer": str(int(val) if val.is_integer() else val),
                    "execution_latency_ms": round(elapsed_ms, 2)
                }

        # 2. Factorials: "Compute the value of 5! (5 factorial)."
        fact_match = re.search(r"(\d+)\s*!\s*(?:\(\d+\s+factorial\))?|factorial\s+of\s+(\d+)", q, re.IGNORECASE)
        if fact_match:
            n = int(fact_match.group(1) or fact_match.group(2))
            val = math.factorial(n)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "success",
                "method": "deterministic_combinatorics (factorial)",
                "parsed_expression": f"{n}!",
                "result": val,
                "formatted_answer": str(val),
                "execution_latency_ms": round(elapsed_ms, 2)
            }

        # 3. Remainder / Modulo: "What is the remainder when 1256 is divided by 9?"
        rem_match = re.search(r"remainder\s+when\s+(\d+)\s+is\s+divided\s+by\s+(\d+)", q, re.IGNORECASE)
        if rem_match:
            a, b = int(rem_match.group(1)), int(rem_match.group(2))
            val = a % b
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "success",
                "method": "deterministic_arithmetic (modulo)",
                "parsed_expression": f"{a} % {b}",
                "result": val,
                "formatted_answer": str(val),
                "execution_latency_ms": round(elapsed_ms, 2)
            }

        # 4. Commercial Discount: "If an item originally costs $250 and is discounted by 20%, what is the final price?"
        disc_match = re.search(r"costs?\s+\$?(\d+(?:\.\d+)?)\s+and\s+is\s+discounted\s+by\s+(\d+(?:\.\d+)?)\s*%", q, re.IGNORECASE)
        if disc_match:
            price = float(disc_match.group(1))
            disc = float(disc_match.group(2)) / 100.0
            val = price * (1.0 - disc)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "success",
                "method": "deterministic_arithmetic (discount)",
                "parsed_expression": f"{price} * (1 - {disc})",
                "result": val,
                "formatted_answer": str(int(val) if val.is_integer() else val),
                "execution_latency_ms": round(elapsed_ms, 2)
            }

        # 5. Algebraic Equations: "Solve for x: 3x - 14 = 49", "Solve the quadratic equation: x^2 - 7x + 12 = 0", "Solve x^2 - 9 = 0"
        eq_clean = re.sub(r"(?i)^solve(?:\s+for\s+[a-zA-Z]|\s+the\s+quadratic\s+equation|\s+equation)?(?::|\s+)", "", q).strip()
        if "=" in eq_clean:
            lhs, rhs = eq_clean.split("=", 1)
            try:
                lhs_expr = parse_expr(lhs.replace("^", "**"), transformations=TRANSFORMATIONS)
                rhs_expr = parse_expr(rhs.replace("^", "**"), transformations=TRANSFORMATIONS)
                equation = sp.Eq(lhs_expr, rhs_expr)
                solutions = sp.solve(equation)
                sol_floats = [float(s) if (hasattr(s, "is_real") and s.is_real) else s for s in solutions]
                sol_str = ", ".join(str(int(s) if (isinstance(s, (int, float)) and s == int(s)) else s) for s in sol_floats)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return {
                    "status": "success",
                    "method": "deterministic_symbolic_solver (SymPy)",
                    "parsed_expression": eq_clean,
                    "result": sol_floats if len(sol_floats) > 1 else (sol_floats[0] if sol_floats else "No solution"),
                    "formatted_answer": f"x = {sol_str}" if sol_str else "No solution found.",
                    "execution_latency_ms": round(elapsed_ms, 2)
                }
            except Exception:
                pass

        # 6. Verbal arithmetic & multi-percentage expressions
        clean_expr = q
        clean_expr = re.sub(r"(?i)^(what\s+is|calculate|compute)\s+", "", clean_expr).rstrip("?. ")
        # Exponents: "28 squared" -> "(28**2)"
        clean_expr = re.sub(r"(\d+(?:\.\d+)?)\s+squared", r"(\1**2)", clean_expr, flags=re.IGNORECASE)
        # Square roots: "square root of 144" -> "sqrt(144)"
        clean_expr = re.sub(r"(?i)\b(?:the\s+)?square\s+root\s+of\s+(\d+(?:\.\d+)?)\b", r"sqrt(\1)", clean_expr)
        clean_expr = re.sub(r"(?i)\bsqrt\s+of\s+(\d+(?:\.\d+)?)\b", r"sqrt(\1)", clean_expr)
        # Percentages: "15% of 840" -> "(0.15 * 840)"
        clean_expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)", r"(\1 / 100 * \2)", clean_expr)
        clean_expr = re.sub(r"\b(times|multiplied\s+by)\b", "*", clean_expr, flags=re.IGNORECASE)
        clean_expr = re.sub(r"\b(plus|added\s+to)\b", "+", clean_expr, flags=re.IGNORECASE)
        clean_expr = re.sub(r"\b(minus|subtracted\s+by)\b", "-", clean_expr, flags=re.IGNORECASE)
        clean_expr = re.sub(r"\b(divided\s+by|over)\b", "/", clean_expr, flags=re.IGNORECASE)

        # Match arithmetic candidates
        try:
            res = parse_expr(clean_expr.replace("^", "**"), transformations=TRANSFORMATIONS)
            eval_res = float(res) if (res.is_real and not res.is_Integer) else (int(res) if res.is_Integer else float(res))
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "success",
                "method": "deterministic_arithmetic_calculator (SymPy)",
                "parsed_expression": clean_expr,
                "result": eval_res,
                "formatted_answer": str(eval_res),
                "execution_latency_ms": round(elapsed_ms, 2)
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "error",
                "method": "math_solver",
                "error": f"Could not parse arithmetic expression: {e}",
                "execution_latency_ms": round(elapsed_ms, 2)
            }
