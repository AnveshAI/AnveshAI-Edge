"""
Advanced Math Engine v2 — symbolic computation using SymPy.

Handles a wide range of advanced mathematics:
    ─ Indefinite & definite integration
    ─ Differentiation (any order, any variable)
    ─ Limits (including one-sided and infinity)
    ─ Equation & system solving
    ─ Ordinary differential equations (ODEs)
    ─ Matrix operations (det, inverse, eigenvalues, rank, trace)
    ─ Taylor / Maclaurin series expansion
    ─ Laplace & inverse Laplace transforms
    ─ Fourier transform
    ─ Simplification, factoring, expansion, partial fractions
    ─ Number theory (GCD, LCM, prime factorization, modular arithmetic)
    ─ Statistics (mean, variance, std deviation, median)
    ─ Combinatorics (factorial, binomial coefficients, permutations)
    ─ Complex number operations
    ─ Summations & products
    ─ Trigonometric identity simplification

The engine parses natural language ("integrate x^2 sin(x)"), runs the
computation symbolically with SymPy, and returns:
    - a clean string result
    - a LaTeX representation

The result is then handed to the LLM, which is TOLD the correct answer
and must only produce the step-by-step explanation — preventing hallucination.
"""

import re
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Operation keyword registry
# ─────────────────────────────────────────────────────────────────────────────

_ADVANCED_OPS: dict[str, list[str]] = {
    "integrate": [
        "integrate", "integral of", "antiderivative of", "indefinite integral",
        "definite integral", "∫",
    ],
    "differentiate": [
        "differentiate", "derivative of", "d/dx", "d/dy", "d/dz", "d/dt",
        "diff of", "first derivative", "second derivative", "third derivative",
        "nth derivative", "partial derivative",
    ],
    "limit": [
        "limit of", "limit as", "lim ", "lim(", "find the limit",
    ],
    "solve": [
        "solve ", "find roots of", "zeros of", "find x such that",
        "find the value of x", "find the solution",
    ],
    "ode": [
        "differential equation", "ode ", "ordinary differential",
        "dsolve", "solve the ode", "solve ode", "y'' ", "y' ",
        "d2y", "d^2y", "solve the differential",
    ],
    "eigenvalue": [
        "eigenvalue", "eigenvector", "eigen value", "eigen vector",
        "characteristic polynomial",
    ],
    "determinant": [
        "determinant of", "det of", "det(",
    ],
    "inverse": [
        "inverse of matrix", "matrix inverse", "inverse matrix",
    ],
    "matrix_rank": [
        "rank of matrix", "matrix rank", "rank(",
    ],
    "matrix_trace": [
        "trace of matrix", "matrix trace", "trace(",
    ],
    "series": [
        "taylor series", "maclaurin series", "series expansion",
        "expand in series", "power series",
    ],
    "laplace": [
        "laplace transform", "laplace of", "l{", "l(",
    ],
    "inverse_laplace": [
        "inverse laplace", "laplace inverse", "l^-1",
    ],
    "fourier": [
        "fourier transform", "fourier of",
    ],
    "simplify": [
        "simplify ", "simplify(", "reduce ",
    ],
    "trig_simplify": [
        "simplify trig", "trig simplif", "trigonometric simplif",
        "simplify the trigonometric",
    ],
    "factor": [
        "factor ", "factorise ", "factorize ", "factorise(", "factor(",
    ],
    "expand": [
        "expand ", "expand(",
    ],
    "partial_fraction": [
        "partial fraction", "partial fractions", "partial fraction decomposition",
    ],
    "gcd": [
        "gcd(", "gcd of", "greatest common divisor", "highest common factor",
        "hcf of",
    ],
    "lcm": [
        "lcm(", "lcm of", "least common multiple", "lowest common multiple",
    ],
    "prime_factors": [
        "prime factor", "prime factorization", "factorise into primes",
        "factorize into primes", "prime decomposition",
    ],
    "modular": [
        " mod ", "modulo ", "modular arithmetic", "modular inverse",
        "congruence",
    ],
    "statistics": [
        "mean of", "average of", "median of", "mode of",
        "variance of", "standard deviation of", "std dev of", "std(",
        "statistics of",
    ],
    "factorial": [
        "factorial of", "factorial(", "! ", "n factorial",
    ],
    "binomial": [
        "binomial coefficient", "choose ", "c(", "combinations of",
        "nCr",
    ],
    "permutation": [
        "permutation", "nPr", "arrangements of",
    ],
    "summation": [
        "sum of ", "summation of", "sigma notation", "∑",
    ],
    "product": [
        "product of ", "∏", "pi product",
    ],
    "complex_ops": [
        "complex number", "real part", "imaginary part", "modulus of",
        "argument of", "conjugate of",
    ],
}


def detect_advanced_operation(text: str) -> Optional[str]:
    """Return the detected advanced math operation (highest-priority match), or None."""
    lowered = text.lower()

    # Priority ordering — more specific ops first
    priority_order = [
        "trig_simplify", "inverse_laplace", "laplace", "fourier",
        "ode", "eigenvalue", "determinant", "inverse", "matrix_rank",
        "matrix_trace", "partial_fraction", "prime_factors", "modular",
        "statistics", "binomial", "permutation", "factorial",
        "summation", "product", "complex_ops", "gcd", "lcm",
        "integrate", "differentiate", "limit", "series",
        "simplify", "factor", "expand", "solve",
    ]

    for op in priority_order:
        keywords = _ADVANCED_OPS.get(op, [])
        for kw in keywords:
            if kw in lowered:
                return op
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Expression helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(expr: str) -> str:
    """Normalise user-written math to SymPy-parseable syntax."""
    expr = expr.strip()
    # Remove trailing differential (dx, dy, dt, …) for integrals
    expr = re.sub(r'\s*d[a-zA-Z]\s*$', '', expr)
    # Remove "= 0" for equation solving — SymPy's solve() takes LHS
    expr = re.sub(r'\s*=\s*0\s*$', '', expr)
    # Replace ^ with **
    expr = expr.replace('^', '**')
    # Natural log → log
    expr = re.sub(r'\bln\b', 'log', expr)
    # arc functions
    expr = re.sub(r'\barc(sin|cos|tan)\b', r'a\1', expr)
    return expr.strip()


def _parse(expr_str: str):
    """
    Parse a string into a SymPy expression.
    Uses implicit multiplication so "x sin(x)" → x*sin(x).
    Raises ValueError on failure.
    """
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    from sympy import symbols
    from sympy import (
        sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
        exp, log, sqrt, pi, E, oo, I, Abs,
        sec, csc, cot, atan2, factorial, binomial,
        ceiling, floor, sign, Heaviside,
    )

    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )

    local_dict = {v: symbols(v) for v in "xyztnkabcmnpqrs"}
    local_dict.update({
        "sin": sin, "cos": cos, "tan": tan,
        "asin": asin, "acos": acos, "atan": atan,
        "arcsin": asin, "arccos": acos, "arctan": atan,
        "sinh": sinh, "cosh": cosh, "tanh": tanh,
        "exp": exp, "log": log, "ln": log,
        "sqrt": sqrt, "pi": pi, "e": E, "E": E,
        "oo": oo, "inf": oo, "infinity": oo,
        "I": I, "j": I, "abs": Abs, "Abs": Abs,
        "sec": sec, "csc": csc, "cot": cot, "atan2": atan2,
        "factorial": factorial, "binomial": binomial,
        "ceil": ceiling, "floor": floor, "sign": sign,
        "Heaviside": Heaviside, "H": Heaviside,
    })

    cleaned = _preprocess(expr_str)
    try:
        return parse_expr(cleaned, local_dict=local_dict,
                          transformations=transformations,
                          evaluate=True)
    except Exception as exc:
        raise ValueError(f"Cannot parse '{expr_str}': {exc}")


def _extract_variable(text: str, default: str = "x") -> str:
    """Detect the primary variable from phrases like 'with respect to y'."""
    m = re.search(r'with\s+respect\s+to\s+([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    m = re.search(r'\bwrt\s+([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    m = re.search(r'\bd/d([a-zA-Z])', text, re.I)
    if m:
        return m.group(1)
    return default


def _strip_prefix(text: str, keywords: list[str]) -> str:
    """Remove any matching operation prefix from the text."""
    lowered = text.lower()
    for kw in sorted(keywords, key=len, reverse=True):
        if lowered.startswith(kw):
            return text[len(kw):].strip()
    for kw in sorted(keywords, key=len, reverse=True):
        idx = lowered.find(kw)
        if idx != -1:
            return text[idx + len(kw):].strip()
    return text.strip()


def _parse_matrix(text: str):
    """Extract and parse a matrix from text like [[1,2],[3,4]]."""
    from sympy import Matrix
    m = re.search(r'\[\[.*?\]\]', text, re.DOTALL)
    if not m:
        raise ValueError(
            "Please provide the matrix in format [[a,b],[c,d]] — e.g. [[1,2],[3,4]]"
        )
    mat_raw = m.group(0)
    mat_data = eval(mat_raw)
    return Matrix(mat_data)


# ─────────────────────────────────────────────────────────────────────────────
# Operation handlers
# ─────────────────────────────────────────────────────────────────────────────

def _handle_integrate(text: str) -> Tuple[str, str]:
    from sympy import integrate, symbols, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["integrate"])
    # Remove "with respect to X" from expression text
    expr_text = re.sub(r'\s+with\s+respect\s+to\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\bwrt\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()

    # Definite integral: "EXPR from A to B"
    m = re.search(
        r'(.*?)\s+from\s+([\w\.\-\+eEpioo∞infty]+)\s+to\s+([\w\.\-\+eEpioo∞infty]+)',
        expr_text, re.I
    )

    def _parse_bound(raw: str):
        raw = raw.replace("infty", "oo").replace("∞", "oo").replace("infinity", "oo")
        import sympy
        if raw == "oo":  return sympy.oo
        if raw == "-oo": return -sympy.oo
        return _parse(raw)

    if m:
        expr   = _parse(m.group(1).strip())
        lower  = _parse_bound(m.group(2).strip())
        upper  = _parse_bound(m.group(3).strip())
        result = integrate(expr, (var, lower, upper))
        return (
            f"∫ ({expr}) d{var_name} from {lower} to {upper} = {result}",
            latex(result),
        )
    else:
        expr   = _parse(expr_text)
        result = integrate(expr, var)
        return (
            f"∫ ({expr}) d{var_name} = {result} + C",
            latex(result) + " + C",
        )


def _handle_differentiate(text: str) -> Tuple[str, str]:
    from sympy import diff, symbols, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)

    _ORDINAL_MAP = {
        "second": 2, "2nd": 2, "third": 3, "3rd": 3,
        "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
        "sixth": 6, "6th": 6, "seventh": 7, "7th": 7,
        "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    }
    order = 1
    m_order = re.search(
        r'\b(second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|'
        r'seventh|7th|eighth|8th|ninth|9th)\s+derivative\b',
        text, re.I
    )
    if m_order:
        order = _ORDINAL_MAP[m_order.group(1).lower()]

    expr_text = text
    expr_text = re.sub(
        r'(?:second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|'
        r'seventh|7th|eighth|8th|ninth|9th)?\s*(?:partial\s+)?derivative\s+of\s+',
        '', expr_text, flags=re.I
    ).strip()
    expr_text = _strip_prefix(expr_text, _ADVANCED_OPS["differentiate"])
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\s+with\s+respect\s+to\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()
    expr_text = re.sub(r'\bwrt\s+[a-zA-Z]\s*$', '', expr_text, flags=re.I).strip()

    expr   = _parse(expr_text)
    result = diff(expr, var, order)
    order_label = {1: "d/d", 2: "d²/d", 3: "d³/d"}.get(order, f"d^{order}/d")
    return (
        f"{order_label}{var_name}[{expr}] = {result}",
        latex(result),
    )


def _handle_limit(text: str) -> Tuple[str, str]:
    from sympy import limit, symbols, latex, oo

    var_name = _extract_variable(text, default="x")
    var = symbols(var_name)

    m = re.search(
        r'(?:limit\s+of\s+|lim\s+)?(.+?)\s+as\s+'
        rf'{var_name}\s+(?:->|→|approaches|tends\s+to)\s+([^\s,]+)',
        text, re.I
    )

    if m:
        expr_raw  = m.group(1).strip()
        point_raw = m.group(2).strip()
    else:
        m2 = re.search(
            rf'lim\s+{var_name}\s*[-→>]{{1,2}}\s*([^\s]+)\s+(.+)', text, re.I
        )
        if m2:
            point_raw = m2.group(1)
            expr_raw  = m2.group(2)
        else:
            raise ValueError(
                "Could not parse limit. Expected: 'limit of EXPR as x approaches VALUE'"
            )

    point_raw = (point_raw.replace("infinity", "oo")
                          .replace("∞", "oo")
                          .replace("infty", "oo"))
    import sympy
    if point_raw == "oo":   point = oo
    elif point_raw == "-oo": point = -oo
    else: point = _parse(point_raw)

    expr   = _parse(expr_raw)
    result = limit(expr, var, point)
    return (
        f"lim({expr}) as {var_name} → {point} = {result}",
        sympy.latex(result),
    )


def _handle_solve(text: str) -> Tuple[str, str]:
    from sympy import solve, symbols, Eq, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["solve"])
    expr_text = re.sub(r'\s+for\s+[a-zA-Z]$', '', expr_text.strip(), flags=re.I)

    if '=' in expr_text:
        parts = expr_text.split('=', 1)
        lhs = _parse(parts[0].strip())
        rhs = _parse(parts[1].strip())
        solutions = solve(Eq(lhs, rhs), var)
    else:
        solutions = solve(_parse(expr_text), var)

    if not solutions:
        return (f"No solutions found for: {expr_text}", r"\text{No solution}")

    sol_str   = ", ".join(str(s) for s in solutions)
    sol_latex = ", ".join(latex(s) for s in solutions)
    return (f"{var_name} = {sol_str}", sol_latex)


def _handle_ode(text: str) -> Tuple[str, str]:
    """Solve ordinary differential equations using SymPy's dsolve."""
    from sympy import symbols, Function, dsolve, latex, Eq, Derivative
    from sympy.parsing.sympy_parser import parse_expr

    x = symbols('x')
    y = Function('y')

    # Normalise ^ to **
    text_norm = text.replace('^', '**')

    # Try to extract the ODE expression:
    # Support patterns like:
    #   "y'' + y = 0", "y' - 2y = 0", "dy/dx + y = x"
    # We'll try to build the ODE equation

    # Replace y'' → Derivative(y(x), x, 2), y' → Derivative(y(x), x)
    # and y → y(x) in the expression
    cleaned = text_norm
    # Strip any leading prompt words
    cleaned = re.sub(
        r'(?:solve|ode|ordinary differential equation|differential equation|solve the ode|solve ode)[\s:]*',
        '', cleaned, flags=re.I
    ).strip()

    # Replace notation
    cleaned = re.sub(r"y''", "Derivative(y(x),x,2)", cleaned)
    cleaned = re.sub(r"y'",  "Derivative(y(x),x)",   cleaned)
    # dy/dx or d^2y/dx^2
    cleaned = re.sub(r'd\*\*2y/dx\*\*2', 'Derivative(y(x),x,2)', cleaned)
    cleaned = re.sub(r'd2y/dx2',          'Derivative(y(x),x,2)', cleaned)
    cleaned = re.sub(r'dy/dx',            'Derivative(y(x),x)',   cleaned)
    # bare y that isn't followed by ( — replace with y(x)
    cleaned = re.sub(r'\by\b(?!\()', 'y(x)', cleaned)

    local_dict = {
        'x': x, 'y': y, 'Derivative': Derivative,
    }
    from sympy import sin, cos, exp, log, sqrt, pi, E, oo, tan
    local_dict.update({
        'sin': sin, 'cos': cos, 'exp': exp, 'log': log,
        'sqrt': sqrt, 'pi': pi, 'e': E, 'tan': tan,
    })

    try:
        if '=' in cleaned:
            lhs_str, rhs_str = cleaned.split('=', 1)
            lhs = parse_expr(lhs_str.strip(), local_dict=local_dict)
            rhs = parse_expr(rhs_str.strip(), local_dict=local_dict)
            ode_eq = Eq(lhs, rhs)
        else:
            expr = parse_expr(cleaned.strip(), local_dict=local_dict)
            ode_eq = Eq(expr, 0)

        sol = dsolve(ode_eq, y(x))
        return (
            f"ODE: {ode_eq}\nGeneral solution: {sol}",
            latex(sol),
        )
    except Exception as exc:
        raise ValueError(f"Could not solve ODE: {exc}")


def _handle_eigenvalue(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    eigs  = mat.eigenvals()
    evecs = mat.eigenvects()

    eig_str = "; ".join(
        f"λ={ev} (multiplicity {mult})" for ev, mult in eigs.items()
    )
    evec_parts = []
    for ev, mult, vecs in evecs:
        for v in vecs:
            evec_parts.append(f"λ={ev}: {v.T.tolist()}")
    evec_str = "; ".join(evec_parts)

    result_str = f"Eigenvalues: {eig_str}\nEigenvectors: {evec_str}"
    return (result_str, eig_str)


def _handle_determinant(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    det = mat.det()
    return (f"det = {det}", latex(det))


def _handle_inverse(text: str) -> Tuple[str, str]:
    from sympy import latex

    mat = _parse_matrix(text)
    inv = mat.inv()
    return (f"Inverse matrix:\n{inv}", latex(inv))


def _handle_matrix_rank(text: str) -> Tuple[str, str]:
    mat = _parse_matrix(text)
    rank = mat.rank()
    return (f"Rank = {rank}", str(rank))


def _handle_matrix_trace(text: str) -> Tuple[str, str]:
    from sympy import latex
    mat = _parse_matrix(text)
    trace = mat.trace()
    return (f"Trace = {trace}", latex(trace))


def _handle_series(text: str) -> Tuple[str, str]:
    from sympy import series, symbols, latex, oo

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["series"])
    # Strip leading "of" left after prefix removal
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()

    point = 0
    m_point = re.search(r'(?:around|at|about|near)\s+([\w\.\-\+]+)', expr_text, re.I)
    if m_point:
        raw = m_point.group(1).replace("infinity", "oo").replace("∞", "oo")
        point = oo if raw == "oo" else _parse(raw)
        expr_text = expr_text[:m_point.start()].strip()

    order = 6
    m_order = re.search(r'(?:order|degree|up\s+to|terms?)\s+(\d+)', expr_text, re.I)
    if m_order:
        order = int(m_order.group(1))
        expr_text = (expr_text[:m_order.start()] + expr_text[m_order.end():]).strip()

    expr   = _parse(expr_text)
    result = series(expr, var, point, n=order)
    return (
        f"Series of {expr} around {var_name}={point} (order {order}): {result}",
        latex(result),
    )


def _handle_laplace(text: str) -> Tuple[str, str]:
    from sympy import symbols, laplace_transform, latex

    t, s = symbols('t s', positive=True)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["laplace"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()

    expr = _parse(expr_text)
    # With noconds=True SymPy returns the expression directly (not a tuple)
    raw = laplace_transform(expr, t, s, noconds=True)
    # Guard: some SymPy versions return a 3-tuple even with noconds=True
    if isinstance(raw, tuple):
        result = raw[0]
    else:
        result = raw
    return (
        f"L{{{expr}}} = {result}",
        latex(result),
    )


def _handle_inverse_laplace(text: str) -> Tuple[str, str]:
    from sympy import symbols, inverse_laplace_transform, latex, Symbol

    # SymPy requires s to be declared positive for inverse Laplace
    t_pos, s_pos = symbols('t s', positive=True)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["inverse_laplace"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()
    expr = _parse(expr_text)
    # Substitute any plain 's' or 't' with the positive versions
    s_plain = Symbol('s')
    t_plain = Symbol('t')
    expr = expr.subs([(s_plain, s_pos), (t_plain, t_pos)])
    result = inverse_laplace_transform(expr, s_pos, t_pos)
    return (
        f"L⁻¹{{{expr}}} = {result}",
        latex(result),
    )


def _handle_fourier(text: str) -> Tuple[str, str]:
    from sympy import symbols, fourier_transform, latex

    x, k = symbols('x k')
    expr_text = _strip_prefix(text, _ADVANCED_OPS["fourier"])
    expr_text = re.sub(r'\bof\b', '', expr_text, flags=re.I).strip()
    expr = _parse(expr_text)
    result = fourier_transform(expr, x, k)
    return (
        f"F{{{expr}}} = {result}",
        latex(result),
    )


def _handle_simplify(text: str) -> Tuple[str, str]:
    from sympy import simplify, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["simplify"])
    expr   = _parse(expr_text)
    result = simplify(expr)
    return (f"Simplified: {result}", latex(result))


def _handle_trig_simplify(text: str) -> Tuple[str, str]:
    from sympy import trigsimp, latex

    # strip any trig-specific prefix then fall through
    expr_text = re.sub(
        r'simplif[y]?\s+(?:the\s+)?trigonometric\s+|trig\s+simplif[y]?\s+|simplif[y]?\s+trig\s+',
        '', text, flags=re.I
    ).strip()
    expr   = _parse(expr_text)
    result = trigsimp(expr)
    return (f"Trig-simplified: {result}", latex(result))


def _handle_factor(text: str) -> Tuple[str, str]:
    from sympy import factor, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["factor"])
    expr   = _parse(expr_text)
    result = factor(expr)
    return (f"Factored: {result}", latex(result))


def _handle_expand(text: str) -> Tuple[str, str]:
    from sympy import expand, latex

    expr_text = _strip_prefix(text, _ADVANCED_OPS["expand"])
    expr   = _parse(expr_text)
    result = expand(expr)
    return (f"Expanded: {result}", latex(result))


def _handle_partial_fraction(text: str) -> Tuple[str, str]:
    from sympy import apart, symbols, latex

    var_name = _extract_variable(text)
    var = symbols(var_name)
    expr_text = _strip_prefix(text, _ADVANCED_OPS["partial_fraction"])
    expr   = _parse(expr_text)
    result = apart(expr, var)
    return (f"Partial fractions of {expr}: {result}", latex(result))


def _handle_gcd(text: str) -> Tuple[str, str]:
    from sympy import gcd, latex

    # Extract numbers from text
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide at least two numbers. Example: GCD of 48 and 18")
    from sympy import Integer
    result = Integer(numbers[0])
    for n in numbers[1:]:
        result = gcd(result, Integer(n))
    nums_str = ", ".join(numbers)
    return (f"GCD({nums_str}) = {result}", latex(result))


def _handle_lcm(text: str) -> Tuple[str, str]:
    from sympy import lcm, latex

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide at least two numbers. Example: LCM of 12 and 18")
    from sympy import Integer
    result = Integer(numbers[0])
    for n in numbers[1:]:
        result = lcm(result, Integer(n))
    nums_str = ", ".join(numbers)
    return (f"LCM({nums_str}) = {result}", latex(result))


def _handle_prime_factors(text: str) -> Tuple[str, str]:
    from sympy import factorint, latex

    numbers = re.findall(r'\d+', text)
    if not numbers:
        raise ValueError("Please provide a number. Example: prime factorization of 360")
    n = int(numbers[0])
    factors = factorint(n)
    factor_str = " × ".join(
        f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items())
    )
    return (f"{n} = {factor_str}", factor_str)


def _handle_modular(text: str) -> Tuple[str, str]:
    from sympy import mod_inverse, Integer

    # modular inverse: "modular inverse of A mod M"
    m_inv = re.search(
        r'modular\s+inverse\s+of\s+(\d+)\s+mod\s+(\d+)', text, re.I
    )
    if m_inv:
        a, m_val = int(m_inv.group(1)), int(m_inv.group(2))
        inv = mod_inverse(a, m_val)
        return (f"Modular inverse of {a} mod {m_val} = {inv}", str(inv))

    # plain modulo: "A mod B"
    m_mod = re.search(r'(\d+)\s+mod(?:ulo)?\s+(\d+)', text, re.I)
    if m_mod:
        a, m_val = int(m_mod.group(1)), int(m_mod.group(2))
        result = a % m_val
        return (f"{a} mod {m_val} = {result}", str(result))

    raise ValueError(
        "Could not parse modular arithmetic. "
        "Try: '17 mod 5' or 'modular inverse of 3 mod 7'"
    )


def _handle_statistics(text: str) -> Tuple[str, str]:
    from sympy.stats import Normal
    from sympy import Rational, latex

    # Extract list of numbers from text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not numbers:
        raise ValueError(
            "Please provide a list of numbers. Example: mean of 2, 4, 6, 8"
        )
    vals = [float(n) for n in numbers]
    n = len(vals)
    mean   = sum(vals) / n
    sorted_vals = sorted(vals)
    if n % 2 == 0:
        median = (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        median = sorted_vals[n//2]
    variance = sum((v - mean) ** 2 for v in vals) / n
    std_dev  = variance ** 0.5

    result_str = (
        f"Data: {vals}\n"
        f"Mean = {mean:.6g}\n"
        f"Median = {median:.6g}\n"
        f"Variance = {variance:.6g}\n"
        f"Std Dev = {std_dev:.6g}"
    )
    return (result_str, result_str.replace("\n", r" \\ "))


def _handle_factorial(text: str) -> Tuple[str, str]:
    from sympy import factorial, latex, Integer

    numbers = re.findall(r'\d+', text)
    if not numbers:
        raise ValueError("Please provide a number. Example: factorial of 10")
    n = int(numbers[0])
    if n > 1000:
        raise ValueError("Number too large for factorial (max 1000)")
    result = factorial(Integer(n))
    return (f"{n}! = {result}", latex(result))


def _handle_binomial(text: str) -> Tuple[str, str]:
    from sympy import binomial as sym_binomial, latex, Integer

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide n and r. Example: binomial coefficient 10 choose 3")
    n, r = int(numbers[0]), int(numbers[1])
    result = sym_binomial(Integer(n), Integer(r))
    return (f"C({n}, {r}) = {result}", latex(result))


def _handle_permutation(text: str) -> Tuple[str, str]:
    from sympy import factorial, latex, Integer

    numbers = re.findall(r'\d+', text)
    if len(numbers) < 2:
        raise ValueError("Please provide n and r. Example: permutation 10 P 3")
    n, r = int(numbers[0]), int(numbers[1])
    result = factorial(Integer(n)) // factorial(Integer(n - r))
    return (f"P({n}, {r}) = {result}", latex(result))


def _handle_summation(text: str) -> Tuple[str, str]:
    from sympy import summation, symbols, oo, latex

    # Try to detect summation variable from "for X=" or "for X from" pattern
    m_var = re.search(r'\bfor\s+([a-zA-Z])\s*(?:=|from)\b', text, re.I)
    if m_var:
        var_name = m_var.group(1)
    else:
        var_name = _extract_variable(text, default="k")
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["summation"])
    # Strip leading "of" left after prefix removal
    expr_text = re.sub(r'^of\s+', '', expr_text, flags=re.I).strip()

    # Pattern A: "EXPR for k=A to B" or "EXPR for k from A to B"
    m = re.search(
        rf'(.*?)\s+for\s+{var_name}\s*(?:=|from)\s*(-?[\w\.]+)\s+to\s+(-?[\w\.∞]+)',
        expr_text, re.I
    )
    # Pattern B: "EXPR from k=A to B"
    if not m:
        m = re.search(
            rf'(.*?)\s+from\s+{var_name}\s*=\s*(-?[\w\.]+)\s+to\s+(-?[\w\.∞]+)',
            expr_text, re.I
        )

    def _parse_bound(raw: str):
        raw = raw.replace("infinity", "oo").replace("∞", "oo").replace("infty", "oo")
        if raw == "oo":  return oo
        if raw == "-oo": return -oo
        return _parse(raw)

    if m:
        expr_raw = m.group(1).strip()
        lo       = _parse_bound(m.group(2))
        hi       = _parse_bound(m.group(3))
        expr     = _parse(expr_raw)
        result   = summation(expr, (var, lo, hi))
        return (
            f"Σ({expr}, {var_name}={lo}..{hi}) = {result}",
            latex(result),
        )
    else:
        expr   = _parse(expr_text)
        result = summation(expr, (var, 0, oo))
        return (
            f"Σ({expr}, {var_name}=0..∞) = {result}",
            latex(result),
        )


def _handle_product(text: str) -> Tuple[str, str]:
    from sympy import Product, symbols, oo, latex

    var_name = _extract_variable(text, default="k")
    var = symbols(var_name)

    expr_text = _strip_prefix(text, _ADVANCED_OPS["product"])
    m = re.search(
        rf'(.*?)\s+(?:for|from)\s+{var_name}\s*=\s*(-?\w+)\s+to\s+(-?\w+)',
        expr_text, re.I
    )
    if m:
        expr_raw = m.group(1).strip()
        lo_raw   = m.group(2).replace("infty", "oo")
        hi_raw   = m.group(3).replace("infty", "oo")
        expr = _parse(expr_raw)
        lo   = oo if lo_raw == "oo" else _parse(lo_raw)
        hi   = oo if hi_raw == "oo" else _parse(hi_raw)
        result = Product(expr, (var, lo, hi)).doit()
        return (
            f"∏({expr}, {var_name}={lo}..{hi}) = {result}",
            latex(result),
        )
    else:
        expr = _parse(expr_text)
        result = Product(expr, (var, 1, oo)).doit()
        return (
            f"∏({expr}, {var_name}=1..∞) = {result}",
            latex(result),
        )


def _handle_complex_ops(text: str) -> Tuple[str, str]:
    from sympy import re as Re, im as Im, Abs, arg, conjugate, latex, symbols, I

    # Try to extract a complex expression
    # Strip common prefixes
    clean = re.sub(
        r'(?:real\s+part\s+of|imaginary\s+part\s+of|modulus\s+of|argument\s+of|conjugate\s+of|complex\s+number)\s*',
        '', text, flags=re.I
    ).strip()

    expr = _parse(clean)

    results = {
        "Real part":      Re(expr),
        "Imaginary part": Im(expr),
        "Modulus":        Abs(expr),
        "Argument":       arg(expr),
        "Conjugate":      conjugate(expr),
    }

    lines = [f"{k} = {v}" for k, v in results.items()]
    result_str  = "\n".join(lines)
    result_latex = r" \\ ".join(f"{k} = {latex(v)}" for k, v in results.items())
    return (result_str, result_latex)


# ─────────────────────────────────────────────────────────────────────────────
# Handler dispatch table
# ─────────────────────────────────────────────────────────────────────────────

_HANDLERS = {
    "integrate":        _handle_integrate,
    "differentiate":    _handle_differentiate,
    "limit":            _handle_limit,
    "solve":            _handle_solve,
    "ode":              _handle_ode,
    "series":           _handle_series,
    "laplace":          _handle_laplace,
    "inverse_laplace":  _handle_inverse_laplace,
    "fourier":          _handle_fourier,
    "simplify":         _handle_simplify,
    "trig_simplify":    _handle_trig_simplify,
    "factor":           _handle_factor,
    "expand":           _handle_expand,
    "partial_fraction": _handle_partial_fraction,
    "eigenvalue":       _handle_eigenvalue,
    "determinant":      _handle_determinant,
    "inverse":          _handle_inverse,
    "matrix_rank":      _handle_matrix_rank,
    "matrix_trace":     _handle_matrix_trace,
    "gcd":              _handle_gcd,
    "lcm":              _handle_lcm,
    "prime_factors":    _handle_prime_factors,
    "modular":          _handle_modular,
    "statistics":       _handle_statistics,
    "factorial":        _handle_factorial,
    "binomial":         _handle_binomial,
    "permutation":      _handle_permutation,
    "summation":        _handle_summation,
    "product":          _handle_product,
    "complex_ops":      _handle_complex_ops,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def solve(user_input: str) -> Tuple[bool, str, str]:
    """
    Main entry point for the advanced math engine.

    Args:
        user_input: Natural language math query.

    Returns:
        (success, result_str, latex_str)
        success    – True if SymPy computed an answer
        result_str – Human-readable answer
        latex_str  – LaTeX of the result
    """
    op = detect_advanced_operation(user_input)
    if op is None:
        return (False, "", "")

    handler = _HANDLERS.get(op)
    if handler is None:
        return (False, f"Operation '{op}' recognised but not yet implemented.", "")

    try:
        result_str, latex_str = handler(user_input)
        return (True, result_str, latex_str)
    except Exception as exc:
        return (False, f"Math engine error ({op}): {exc}", "")
