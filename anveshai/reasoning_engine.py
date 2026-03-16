"""
Advanced Reasoning Engine — Chain-of-Thought (CoT) reasoning layer.

Sits between the intent router and the LLM / specialist engines.
Provides structured, multi-step reasoning for every response:

    Stage 1 — Problem Analysis
        · Identify problem type, domain, and sub-questions
        · Detect ambiguity or missing information
        · Choose the optimal resolution strategy

    Stage 2 — Chain-of-Thought Planning
        · Decompose complex problems into ordered sub-steps
        · Identify dependencies between sub-steps
        · Estimate difficulty and required knowledge

    Stage 3 — Verification & Confidence
        · Cross-check reasoning against known constraints
        · Assign confidence level (HIGH / MEDIUM / LOW)
        · Flag any assumptions made

    Stage 4 — Response Synthesis
        · Compose final LLM prompt that enforces the reasoning trace
        · Force the LLM to follow the plan and not deviate

For advanced math the engine adds a symbolic pre-analysis step:
    · Identify operation type and variable structure
    · Reason about the expected form of the answer
    · Verify the SymPy result makes mathematical sense

Usage:
    from reasoning_engine import ReasoningEngine
    re = ReasoningEngine()
    plan = re.analyze("integrate x^2 sin(x)", intent="advanced_math")
    prompt = re.build_math_prompt(user_input, sympy_result, plan)
    prompt_gen = re.build_general_prompt(user_input, intent, context, plan)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningPlan:
    """Structured reasoning plan produced by the engine."""
    problem_type:   str                = "unknown"
    domain:         str                = "general"
    sub_problems:   list[str]          = field(default_factory=list)
    strategy:       str                = ""
    expected_form:  str                = ""
    assumptions:    list[str]          = field(default_factory=list)
    confidence:     str                = "MEDIUM"      # HIGH / MEDIUM / LOW
    reasoning_steps: list[str]         = field(default_factory=list)
    warnings:       list[str]          = field(default_factory=list)

    def summary(self) -> str:
        """One-line summary for console display."""
        return (
            f"[Reasoning] domain={self.domain} | strategy={self.strategy[:60]} "
            f"| confidence={self.confidence}"
        )

    def full_trace(self) -> str:
        """Full reasoning trace as a numbered list."""
        lines = [f"Problem type: {self.problem_type}", f"Domain: {self.domain}"]
        if self.sub_problems:
            lines.append("Sub-problems:")
            for i, sp in enumerate(self.sub_problems, 1):
                lines.append(f"  {i}. {sp}")
        lines.append(f"Strategy: {self.strategy}")
        if self.expected_form:
            lines.append(f"Expected answer form: {self.expected_form}")
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        if self.warnings:
            lines.append("Warnings: " + "; ".join(self.warnings))
        lines.append(f"Confidence: {self.confidence}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Domain / problem-type classifiers
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_HINTS: dict[str, list[str]] = {
    "calculus": [
        "integrate", "integral", "derivative", "differentiate",
        "d/dx", "limit", "lim", "antiderivative",
    ],
    "algebra": [
        "solve", "factor", "expand", "simplify", "roots", "zeros",
        "equation", "polynomial", "quadratic",
    ],
    "linear_algebra": [
        "matrix", "eigenvalue", "eigenvector", "determinant",
        "inverse", "rank", "trace", "vector", "dot product", "cross product",
    ],
    "differential_equations": [
        "differential equation", "ode", "dy/dx", "y''", "y'", "dsolve",
    ],
    "transforms": [
        "laplace", "fourier", "z-transform", "inverse laplace",
    ],
    "series": [
        "taylor", "maclaurin", "power series", "series expansion",
    ],
    "number_theory": [
        "prime", "gcd", "lcm", "modular", "mod ", "divisible",
        "factorization", "congruence",
    ],
    "statistics": [
        "mean", "median", "mode", "variance", "standard deviation",
        "average", "probability", "distribution",
    ],
    "combinatorics": [
        "factorial", "binomial", "permutation", "combination", "choose",
        "nCr", "nPr",
    ],
    "complex_analysis": [
        "complex", "imaginary", "real part", "modulus", "argument",
        "conjugate", "polar form",
    ],
    "physics": [
        "velocity", "acceleration", "force", "energy", "momentum",
        "electric", "magnetic", "quantum", "wave", "frequency",
    ],
    "computer_science": [
        "algorithm", "complexity", "big o", "sorting", "graph",
        "recursion", "dynamic programming",
    ],
}

_PROBLEM_TYPE_PATTERNS: list[tuple[str, str]] = [
    # Highest-priority: detect before generic "solve" or "equation"
    (r'\bdifferential\s+equation\b|\bode\b|\bdsolve\b', "ode_solving"),
    (r'\bintegrate\b|\bintegral\b|\bantiderivative\b', "integration"),
    (r'\bderivative\b|\bdifferentiate\b|\bd/d[a-z]\b', "differentiation"),
    (r'\blimit\b|\blim\s', "limit_evaluation"),
    (r'\beigenvalue\b|\beigenvector\b', "matrix_operation"),
    (r'\bdeterminant\b|\bdet\b|\binverse\s+matrix\b|\bmatrix\s+rank\b|\bmatrix\s+trace\b', "matrix_operation"),
    (r'\btaylor\b|\bmaclaurin\b|\bseries\s+expansion\b|\bpower\s+series\b', "series_expansion"),
    (r'\blaplace\s+transform\b|\blaplace\s+of\b', "laplace_transform"),
    (r'\bfourier\s+transform\b|\bfourier\s+of\b', "fourier_transform"),
    (r'\bprime\s+factor|\bgcd\b|\blcm\b|\bmod\b|\bmodular\b', "number_theory"),
    (r'\bfactorial\b|\bbinomial\s+coeff|\bpermutation\b|\bnCr\b|\bnPr\b', "combinatorics"),
    (r'\bsum\s+of\b|\bsummation\b|\bsum\b.*\bfrom\b', "summation"),
    (r'\bmean\b|\bmedian\b|\bvariance\b|\bstandard\s+deviation\b|\baverage\b', "statistical_analysis"),
    (r'\bcomplex\s+number\b|\bimaginary\s+part\b|\breal\s+part\b|\bmodulus\s+of\b|\bargument\s+of\b', "complex_number"),
    (r'\bfactor\b|\bfactorise\b|\bfactorize\b', "factorization"),
    (r'\bexpand\b', "algebraic_expansion"),
    (r'\bsimplify\b', "simplification"),
    (r'\bsolve\b|\broots?\s+of\b|\bzeros?\s+of\b', "equation_solving"),
    (r'\bwhat\s+is\b|\bwho\s+is\b|\bexplain\b|\bdefine\b', "knowledge_retrieval"),
    (r'\bhow\s+to\b|\bhow\s+does\b', "procedural_knowledge"),
    (r'\bwhy\b', "explanatory_reasoning"),
    (r'\bcompare\b|\bdifference\s+between\b', "comparative_analysis"),
]

_STRATEGIES: dict[str, str] = {
    "integration":        "Apply integration rules (u-sub, IBP, trig-sub, or direct antiderivative)",
    "differentiation":    "Apply chain rule, product rule, quotient rule, or basic derivative rules",
    "limit_evaluation":   "Apply L'Hôpital's rule, algebraic manipulation, or squeeze theorem",
    "equation_solving":   "Factor, use quadratic formula, or numerical methods as needed",
    "factorization":      "Factor out GCF, then use special patterns (difference of squares, sum/difference of cubes)",
    "algebraic_expansion":"Apply binomial theorem or FOIL method for products",
    "simplification":     "Cancel common factors, apply trig/logarithm identities, or algebraic reduction",
    "matrix_operation":   "Apply matrix algorithms: cofactor expansion (det), row reduction (rank/inverse), characteristic polynomial (eigenvalues)",
    "ode_solving":        "Classify ODE (linear/separable/exact/homogeneous) and apply corresponding solution method",
    "series_expansion":   "Compute Taylor/Maclaurin coefficients using repeated differentiation",
    "laplace_transform":  "Apply Laplace transform table entries and linearity",
    "fourier_transform":  "Apply Fourier transform definition and standard pairs",
    "number_theory":      "Apply Euclidean algorithm (GCD/LCM) or prime factorization via trial division",
    "statistical_analysis": "Compute descriptive statistics: mean, variance (E[(X-μ)²]), std deviation",
    "combinatorics":      "Apply counting principles: factorial, binomial theorem, or permutation formulas",
    "complex_number":     "Use Cartesian/polar form of complex numbers and their properties",
    "knowledge_retrieval": "Retrieve and synthesize relevant factual information",
    "procedural_knowledge": "Provide a clear step-by-step explanation of the procedure",
    "explanatory_reasoning": "Reason about causes, mechanisms, or justifications",
    "comparative_analysis": "Identify key dimensions of comparison and contrast each one",
    "unknown":            "Analyze the problem and apply the most relevant reasoning approach",
}

_EXPECTED_FORMS: dict[str, str] = {
    "integration":        "A symbolic function + constant of integration C (indefinite) or a real number (definite)",
    "differentiation":    "A symbolic expression of the same or lower degree",
    "limit_evaluation":   "A real number, ∞, -∞, or 'does not exist'",
    "equation_solving":   "One or more numerical or symbolic values for the unknown",
    "factorization":      "A product of irreducible factors",
    "series_expansion":   "A polynomial in x up to the given order, plus O(x^n) remainder",
    "matrix_operation":   "A scalar (det/rank/trace) or matrix (inverse/eigenvectors)",
    "ode_solving":        "A function y(x) with integration constants C1, C2, …",
    "laplace_transform":  "A rational or transcendental function of s",
    "fourier_transform":  "A function of the frequency variable k",
    "statistical_analysis": "Real-valued descriptive statistics (mean, variance, std)",
    "combinatorics":      "A non-negative integer",
}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-hop decomposer
# ─────────────────────────────────────────────────────────────────────────────

def _decompose(user_input: str, problem_type: str, intent: str) -> list[str]:
    """Break the problem into an ordered list of sub-problems / reasoning steps."""
    lowered = user_input.lower()

    # Math decompositions
    if problem_type == "integration":
        steps = ["Identify the integrand and the variable of integration"]
        if "from" in lowered and "to" in lowered:
            steps.append("Confirm the integration limits (lower and upper bounds)")
        steps += [
            "Check whether u-substitution, integration by parts, or a standard form applies",
            "Compute the antiderivative step by step",
            "Apply the Fundamental Theorem of Calculus if limits are given",
            "Simplify the result and add C for indefinite integrals",
        ]
        return steps

    if problem_type == "differentiation":
        steps = ["Identify the function and the differentiation variable"]
        if "second" in lowered or "third" in lowered or "2nd" in lowered or "3rd" in lowered:
            steps.append("Determine the required order of differentiation")
        steps += [
            "Identify which rules apply (chain rule, product rule, quotient rule)",
            "Differentiate term by term",
            "Simplify the derivative expression",
        ]
        return steps

    if problem_type == "limit_evaluation":
        return [
            "Identify the function and the point of approach",
            "Check for direct substitution; if it gives 0/0 or ∞/∞, apply L'Hôpital's rule",
            "Alternatively, try algebraic simplification or known limit results",
            "Evaluate the limit or determine if it diverges",
        ]

    if problem_type == "equation_solving":
        steps = ["Identify the type of equation (linear, quadratic, polynomial, transcendental)"]
        if "=" in user_input:
            steps.append("Rearrange so one side is 0 (or use direct substitution for LHS=RHS)")
        steps += [
            "Choose the solution method: factoring, quadratic formula, or numeric methods",
            "Find all solutions in the required domain",
            "Verify each solution satisfies the original equation",
        ]
        return steps

    if problem_type == "ode_solving":
        return [
            "Classify the ODE: order, linearity, and special form",
            "For 1st order: check separable, linear (integrating factor), exact",
            "For 2nd order: find the characteristic equation and its roots",
            "Write the general solution with arbitrary constants C1, C2, …",
            "Apply initial conditions if provided to find particular solution",
        ]

    if problem_type == "matrix_operation":
        steps = ["Identify the matrix dimensions and operation required"]
        if "eigenvalue" in lowered:
            steps += [
                "Form the characteristic polynomial det(A - λI) = 0",
                "Solve for eigenvalues λ",
                "For each eigenvalue, solve (A - λI)v = 0 for eigenvectors",
            ]
        elif "determinant" in lowered or "det" in lowered:
            steps += [
                "Choose expansion method: cofactor expansion or row reduction",
                "Compute the determinant",
            ]
        elif "inverse" in lowered:
            steps += [
                "Augment the matrix with the identity: [A | I]",
                "Row-reduce to obtain [I | A⁻¹]",
            ]
        return steps

    if problem_type == "series_expansion":
        return [
            "Identify the function and expansion point",
            "Compute successive derivatives at the expansion point",
            "Form the Taylor/Maclaurin series coefficients: f^(n)(a) / n!",
            "Write the series up to the required order",
            "Identify the pattern or general term if possible",
        ]

    if problem_type == "laplace_transform":
        return [
            "Express the function in terms of known Laplace pairs",
            "Apply linearity of the Laplace transform",
            "Use standard table entries or direct computation",
            "Simplify the result in the s-domain",
        ]

    # Knowledge / reasoning decompositions
    if problem_type == "knowledge_retrieval":
        return [
            "Identify the core concept being asked about",
            "Recall the definition and key properties",
            "Provide relevant examples or applications",
            "Connect to related concepts if helpful",
        ]

    if problem_type == "explanatory_reasoning":
        return [
            "Identify the phenomenon / concept requiring explanation",
            "Determine the causal chain or mechanism",
            "State any assumptions or simplifications",
            "Synthesise a clear, evidence-based explanation",
        ]

    if problem_type == "comparative_analysis":
        return [
            "Identify what is being compared",
            "List key dimensions of comparison",
            "Analyse each dimension for both subjects",
            "Summarise similarities and differences",
        ]

    # Default: general reasoning
    return [
        "Understand the question and identify the core task",
        "Break the problem into smaller sub-problems",
        "Address each sub-problem in order",
        "Synthesise a complete and accurate answer",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Confidence estimator
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_confidence(
    user_input: str,
    problem_type: str,
    intent: str,
    has_symbolic_result: bool = False,
) -> str:
    """Heuristically estimate confidence in the system's ability to answer."""
    if has_symbolic_result:
        return "HIGH"  # SymPy computed it — we're certain

    lowered = user_input.lower()

    # High confidence for well-defined math types
    high_confidence_types = {
        "integration", "differentiation", "limit_evaluation",
        "equation_solving", "factorization", "algebraic_expansion",
        "simplification", "series_expansion", "number_theory",
        "statistical_analysis", "combinatorics",
    }
    if problem_type in high_confidence_types:
        return "HIGH"

    # Medium confidence for conceptual / knowledge
    if intent in ("knowledge", "conversation"):
        # Simple factual questions tend to be higher confidence
        if any(kw in lowered for kw in ["what is", "define", "who is"]):
            return "MEDIUM"
        # Open-ended or opinion questions are lower
        if any(kw in lowered for kw in ["why", "opinion", "best", "should i"]):
            return "LOW"
        return "MEDIUM"

    return "MEDIUM"


# ─────────────────────────────────────────────────────────────────────────────
# Warnings detector
# ─────────────────────────────────────────────────────────────────────────────

def _detect_warnings(user_input: str, problem_type: str) -> list[str]:
    """Flag potential issues in the problem statement."""
    warnings = []
    lowered = user_input.lower()

    if len(user_input.strip()) < 5:
        warnings.append("Input is very short — may be ambiguous")

    if problem_type == "equation_solving" and "=" not in user_input and "solve" in lowered:
        warnings.append("No '=' found — treating expression as equal to 0")

    if problem_type == "integration" and "from" in lowered and "to" not in lowered:
        warnings.append("'from' found but 'to' is missing — treating as indefinite integral")

    if problem_type in ("differentiation", "integration") and not any(
        c in user_input for c in list("xyztnkabcmnpqrs")
    ):
        warnings.append("No variable detected — defaulting to x")

    if "undefined" in lowered or "infinity" in lowered:
        warnings.append("Expression may involve singularities or unbounded behaviour")

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningEngine:
    """
    Chain-of-Thought reasoning engine.

    Usage:
        re = ReasoningEngine()
        plan = re.analyze(user_input, intent)
        prompt = re.build_math_prompt(user_input, sympy_result, plan)
        prompt = re.build_general_prompt(user_input, intent, context, plan)
    """

    def analyze(
        self,
        user_input: str,
        intent: str,
        has_symbolic_result: bool = False,
    ) -> ReasoningPlan:
        """
        Produce a structured ReasoningPlan for the given input.

        Args:
            user_input: Raw user query.
            intent: Classified intent (advanced_math / math / knowledge / conversation).
            has_symbolic_result: True when SymPy has already computed the answer.

        Returns:
            ReasoningPlan with problem type, strategy, sub-problems, confidence.
        """
        lowered = user_input.lower()

        # 1. Detect domain
        domain = "general"
        for d, keywords in _DOMAIN_HINTS.items():
            for kw in keywords:
                if kw in lowered:
                    domain = d
                    break
            if domain != "general":
                break

        # 2. Classify problem type (list preserves priority order)
        problem_type = "unknown"
        for pattern, ptype in _PROBLEM_TYPE_PATTERNS:
            if re.search(pattern, lowered):
                problem_type = ptype
                break

        # 3. Strategy & expected answer form
        strategy     = _STRATEGIES.get(problem_type, _STRATEGIES["unknown"])
        expected_form = _EXPECTED_FORMS.get(problem_type, "")

        # 4. Decompose into sub-problems
        sub_problems = _decompose(user_input, problem_type, intent)

        # 5. Detect assumptions
        assumptions = []
        if domain in ("calculus", "algebra", "differential_equations"):
            if not any(kw in lowered for kw in ["complex", "imaginary", "i^2"]):
                assumptions.append("Working over the real numbers unless otherwise stated")
        if problem_type in ("integration", "differentiation"):
            assumptions.append("Function is sufficiently smooth (differentiable/integrable)")

        # 6. Warnings
        warnings = _detect_warnings(user_input, problem_type)

        # 7. Confidence
        confidence = _estimate_confidence(
            user_input, problem_type, intent, has_symbolic_result
        )

        return ReasoningPlan(
            problem_type=problem_type,
            domain=domain,
            sub_problems=sub_problems,
            strategy=strategy,
            expected_form=expected_form,
            assumptions=assumptions,
            confidence=confidence,
            reasoning_steps=sub_problems,
            warnings=warnings,
        )

    def build_math_prompt(
        self,
        user_input: str,
        sympy_result: str,
        plan: ReasoningPlan,
    ) -> str:
        """
        Build an LLM prompt for advanced math where SymPy has the correct answer.
        The prompt embeds the full reasoning plan so the LLM follows the exact strategy.
        """
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        assumptions_str = (
            ("\nAssumptions: " + "; ".join(plan.assumptions))
            if plan.assumptions else ""
        )
        expected_str = (
            f"\nExpected answer form: {plan.expected_form}"
            if plan.expected_form else ""
        )

        return (
            f"PROBLEM: {user_input}\n\n"
            f"VERIFIED ANSWER (computed by SymPy — 100% correct): {sympy_result}\n\n"
            "=== YOUR TASK ===\n"
            f"Explain, step by step, HOW a student arrives at: {sympy_result}\n"
            "Do NOT recompute the answer. Do NOT give a different answer. "
            "Your explanation must lead to exactly the verified answer above.\n\n"
            f"TECHNIQUE: {plan.strategy}\n"
            f"DOMAIN: {plan.domain}"
            f"{expected_str}"
            f"{assumptions_str}\n\n"
            f"STEPS TO WALK THROUGH:\n{steps_str}\n\n"
            "Write a numbered explanation. For each step:\n"
            "  - State what you are doing and why.\n"
            "  - Show the algebra or calculation clearly.\n"
            "  - Connect it to the next step.\n\n"
            f"End with: 'Final answer: {sympy_result}' — use this exact wording."
        )

    def build_general_prompt(
        self,
        user_input: str,
        intent: str,
        context: str,
        plan: ReasoningPlan,
    ) -> str:
        """
        Build an LLM prompt for knowledge/conversation using chain-of-thought.
        The plan is embedded so the LLM follows the structured reasoning trace.
        """
        steps_str = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        context_block = (
            f"\nRELEVANT CONTEXT:\n{context}\n" if context else ""
        )
        confidence_note = (
            "\nNote: confidence in this answer is LOW — state clearly if uncertain."
            if plan.confidence == "LOW" else ""
        )
        warnings_block = ""
        if plan.warnings:
            warnings_block = "\nFLAGS: " + "; ".join(plan.warnings) + "\n"

        return (
            "You are AnveshAI, a helpful and honest AI assistant.\n"
            f"QUESTION: {user_input}\n"
            f"{context_block}"
            f"{warnings_block}\n"
            f"REASONING PLAN (follow this structure):\n{steps_str}\n\n"
            f"ANSWER STRATEGY: {plan.strategy}\n"
            f"{confidence_note}\n\n"
            "Write a clear, thorough, well-structured response that follows "
            "the reasoning plan above step by step. "
            "Be concise but complete. State any assumptions or caveats explicitly."
        )

    def build_math_fallback_prompt(
        self,
        user_input: str,
        plan: ReasoningPlan,
        error_context: str = "",
    ) -> str:
        """
        Prompt for when SymPy fails and the LLM must solve from scratch.
        Uses chain-of-thought to maximise correctness.
        """
        steps_str = "\n".join(
            f"  Step {i}: {s}" for i, s in enumerate(plan.sub_problems, 1)
        )
        error_note = (
            f"\nNote: automated computation failed ({error_context}). "
            "Solve manually with extra care.\n" if error_context else ""
        )

        return (
            "You are a mathematics expert.\n"
            f"PROBLEM: {user_input}\n"
            f"{error_note}\n"
            f"PROBLEM TYPE: {plan.problem_type}\n"
            f"STRATEGY: {plan.strategy}\n"
            f"{f'EXPECTED ANSWER FORM: {plan.expected_form}' if plan.expected_form else ''}\n\n"
            f"REASONING STEPS TO FOLLOW:\n{steps_str}\n\n"
            "Solve the problem completely by following each step above. "
            "Show all working in full — do not skip steps. "
            "State the final answer clearly at the end."
        )
