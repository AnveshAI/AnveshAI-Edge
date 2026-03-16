"""
Math Engine — safely evaluates mathematical expressions.

Supported operations:
    +  Addition
    -  Subtraction
    *  Multiplication
    /  Division
    %  Modulo
    ^  Exponentiation (converted to **)
    () Parentheses for grouping

Uses a whitelist-based safe evaluator to avoid arbitrary code execution.
"""

import re
import ast
import operator


# Allowed AST node types for safe evaluation
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,    # unary minus, e.g. -5
    ast.UAdd,    # unary plus, e.g. +5
)

# Mapping AST operators to Python functions
OPERATORS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Mod:  operator.mod,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Prefixes the user might add before the expression
STRIP_PREFIXES = (
    "calculate", "compute", "evaluate", "solve",
    "what is", "what's",
)


def extract_expression(user_input: str) -> str:
    """
    Strip any natural-language prefixes to get the raw math expression.
    Also converts '^' to '**' for exponentiation.
    """
    text = user_input.strip()
    lowered = text.lower()

    for prefix in STRIP_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # Replace caret with Python exponentiation operator
    text = text.replace("^", "**")

    return text


def safe_eval(expression: str) -> float:
    """
    Safely evaluate a mathematical expression string.

    Raises ValueError if the expression is invalid or contains disallowed
    operations.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError(f"Invalid mathematical expression: '{expression}'")

    # Validate every node in the AST
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(
                f"Unsafe or unsupported operation detected: {type(node).__name__}"
            )

    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_func = OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("Division by zero is not allowed.")
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_func = OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(operand)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def evaluate(user_input: str) -> str:
    """
    Main entry point: extract the expression, evaluate it, and return
    a formatted result string.
    """
    expression = extract_expression(user_input)

    if not expression:
        return "Please provide a mathematical expression to evaluate."

    try:
        result = safe_eval(expression)

        # Format: show integer if result is whole, else show decimal
        if isinstance(result, float) and result.is_integer():
            formatted = str(int(result))
        else:
            formatted = f"{result:.6g}"  # up to 6 significant figures

        return f"The result of {expression} = {formatted}"

    except ValueError as e:
        return f"Math error: {e}"
    except Exception as e:
        return f"Unexpected error evaluating expression: {e}"
