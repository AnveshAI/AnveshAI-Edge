"""
Intent Router — classifies user input into one of five categories:
    - system        : /commands
    - advanced_math : calculus, algebra, limits, matrices, series, etc.
    - math          : plain arithmetic (numbers only)
    - knowledge     : factual questions answered from the knowledge base
    - conversation  : everything else (chat, small talk, open questions)
"""

import re


# ── Advanced math detection ───────────────────────────────────────────────────
# These keywords signal symbolic / advanced mathematics.
# Checked BEFORE the simple-arithmetic detector.
ADVANCED_MATH_KEYWORDS = [
    # Calculus
    "integrate", "integral", "antiderivative", "indefinite integral",
    "definite integral", "∫",
    "differentiate", "derivative", "d/dx", "d/dy", "d/dz", "d/dt",
    "second derivative", "third derivative", "nth derivative", "partial derivative",
    "limit of", "limit as", "lim ", "find the limit",
    # Differential equations
    "differential equation", "ode ", "dsolve", "solve the ode",
    "y'' ", "y' ", "d2y", "d^2y",
    # Algebra / equations
    "solve ", "find roots", "zeros of", "find the value of x",
    # Linear algebra
    "eigenvalue", "eigenvector", "determinant of", "det of",
    "inverse matrix", "matrix inverse", "rank of matrix", "matrix rank",
    "trace of matrix", "matrix trace", "characteristic polynomial",
    # Series & transforms
    "taylor series", "maclaurin series", "series expansion", "power series",
    "laplace transform", "laplace of", "inverse laplace",
    "fourier transform", "fourier of",
    # Symbolic manipulation
    "simplify ", "simplify(", "factor ", "factorise", "factorize",
    "expand ", "partial fraction",
    # Number theory
    "gcd(", "gcd of", "greatest common divisor", "highest common factor",
    "lcm(", "lcm of", "least common multiple",
    "prime factor", "prime factorization",
    " mod ", "modulo ", "modular inverse",
    # Statistics
    "mean of", "average of", "median of", "variance of",
    "standard deviation of", "std dev of",
    # Combinatorics
    "factorial of", "factorial(", "binomial coefficient",
    "choose ", "nCr", "nPr", "permutation",
    # Summations & products
    "sum of ", "summation of", "∑", "∏",
    # Complex numbers
    "complex number", "real part of", "imaginary part of",
    "modulus of", "argument of", "conjugate of",
    # Trigonometric identity simplification
    "simplify trig", "trig simplif", "trigonometric simplif",
]

# ── Simple arithmetic detection ───────────────────────────────────────────────
MATH_PATTERN = re.compile(
    r"""
    ^           # start of string
    \s*         # optional leading whitespace
    [\d\s\(\)]  # starts with digit, space, or parenthesis
    [\d\s\+\-\*\/\%\^\(\)\.]*  # followed by math characters
    $           # end of string
    """,
    re.VERBOSE,
)

ARITHMETIC_PREFIXES = ("calculate", "compute", "evaluate", "what is")

# ── Knowledge detection ───────────────────────────────────────────────────────
KNOWLEDGE_KEYWORDS = [
    "what is", "what are", "who is", "who are", "explain", "define",
    "tell me about", "describe", "how does", "why is", "when was",
    "where is", "history of", "meaning of", "difference between",
    "knowledge", "information about", "learn about", "facts about",
]

# ── System commands ───────────────────────────────────────────────────────────
SYSTEM_PATTERN = re.compile(r"^/\w+")


def classify_intent(user_input: str) -> str:
    """
    Classify user input and return one of:
        'system' | 'advanced_math' | 'math' | 'knowledge' | 'conversation'
    """
    text   = user_input.strip()
    lowered = text.lower()

    # 1. System commands (/exit, /help, /history)
    if SYSTEM_PATTERN.match(text):
        return "system"

    # 2. Advanced math — symbolic operations (checked before simple math)
    for kw in ADVANCED_MATH_KEYWORDS:
        if kw in lowered:
            return "advanced_math"

    # 3. Simple arithmetic — numbers and operators only
    if _is_simple_arithmetic(text, lowered):
        return "math"

    # 4. Knowledge questions
    for kw in KNOWLEDGE_KEYWORDS:
        if kw in lowered:
            return "knowledge"

    # 5. Fallback — conversation / LLM
    return "conversation"


def _is_simple_arithmetic(text: str, lowered: str) -> bool:
    """True if the input is a plain numeric arithmetic expression."""
    # Allow optional natural-language prefix before checking expression
    remainder = lowered
    for prefix in ARITHMETIC_PREFIXES:
        if lowered.startswith(prefix):
            remainder = lowered[len(prefix):].strip()
            break

    # Must have at least one digit and one operator
    has_digit    = any(ch.isdigit() for ch in remainder)
    has_operator = any(ch in "+-*/%^" for ch in remainder)
    # Must NOT contain letters beyond simple operators (no variables)
    has_letters  = bool(re.search(r'[a-zA-Z]', remainder))

    if has_digit and has_operator and not has_letters:
        return True

    # Also match pure numeric patterns via regex
    return bool(MATH_PATTERN.match(text))
