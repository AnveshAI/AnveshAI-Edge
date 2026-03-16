"""
AnveshAI Edge Prototype
=======================
Terminal-based offline AI assistant — hierarchical modular architecture
with chain-of-thought (CoT) reasoning.

Routing & reasoning chain:
    /commands        →  inline handler                       (instant)
    Plain arithmetic →  math_engine                         (instant, AST-based)
    Advanced math    →  ReasoningEngine.analyze()           (problem decomposition)
                          └─ advanced_math_engine           (SymPy symbolic)
                               └─ ReasoningEngine.build_math_prompt()
                                    → LLM step-by-step explanation
    Knowledge query  →  ReasoningEngine.analyze()           (CoT planning)
                          └─ knowledge_engine               (local KB)
                               ├─ match found → return KB paragraph
                               └─ no match   → ReasoningEngine.build_general_prompt()
                                               → LLM structured answer
    Conversation     →  conversation_engine                 (pattern rules)
                          └─ no pattern → ReasoningEngine.build_general_prompt()
                                           → LLM structured answer

Commands:
    /help    → list commands
    /history → last 10 interactions
    /exit    → quit
"""

import sys

# ── Optional colour support ───────────────────────────────────────────────────
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
except ImportError:
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore = Style = _NoColor()

# ── Module imports ────────────────────────────────────────────────────────────
from router                import classify_intent
from math_engine           import evaluate as math_evaluate
from advanced_math_engine  import solve as advanced_math_solve
from knowledge_engine      import KnowledgeEngine
from conversation_engine   import ConversationEngine
from llm_engine            import LLMEngine, MATH_SYSTEM_PROMPT, MATH_TEMPERATURE
from reasoning_engine      import ReasoningEngine
from memory                import initialize_db, save_interaction, format_history


# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════╗
║     ___                    __   ___   ____           ║
║    / _ | ___ _  _____ ___ / /  / _ | /  _/           ║
║   / __ |/ _ \ |/ / -_|_-</ _ \/ __ |_/ /             ║
║  /_/ |_/_//_/___/\__/___/_//_/_/ |_/___/  EDGE       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
  Available commands:
    /help     — show this help message
    /history  — display last 10 conversation entries
    /exit     — quit AnveshAI Edge

  How to use:
    • Advanced math  →  symbolic engine computes the EXACT answer,
                        LLM explains step-by-step working

      Calculus:
        "integrate x^2 sin(x)"
        "definite integral of x^2 from 0 to 3"
        "derivative of x^3 + 2x"
        "second derivative of sin(x) * e^x"
        "limit of sin(x)/x as x approaches 0"

      Algebra & equations:
        "solve x^2 - 5x + 6 = 0"
        "solve 2x + 3 = 7"

      Differential equations:
        "solve differential equation y'' + y = 0"
        "solve ode dy/dx = y"

      Series & transforms:
        "taylor series of e^x around 0 order 6"
        "laplace transform of sin(t)"
        "inverse laplace of 1/(s^2 + 1)"
        "fourier transform of exp(-x^2)"

      Matrices:
        "determinant of [[1,2],[3,4]]"
        "inverse matrix [[2,1],[5,3]]"
        "eigenvalue [[4,1],[2,3]]"
        "rank of matrix [[1,2,3],[4,5,6]]"

      Symbolic manipulation:
        "factor x^3 - 8"
        "simplify (x^2 - 1)/(x - 1)"
        "expand (x + y)^4"
        "partial fraction 1/(x^2 - 1)"

      Number theory:
        "gcd of 48 and 18"
        "lcm of 12 and 15"
        "prime factorization of 360"
        "17 mod 5"
        "modular inverse of 3 mod 7"

      Statistics:
        "mean of 2, 4, 6, 8, 10"
        "standard deviation of 1, 2, 3, 4, 5"

      Combinatorics:
        "factorial of 10"
        "binomial coefficient 10 choose 3"
        "permutation 6 P 2"

      Summations:
        "sum of k^2 for k from 1 to 10"
        "summation of 1/n^2 for n from 1 to infinity"

      Complex numbers:
        "real part of 3 + 4*I"
        "modulus of 3 + 4*I"

    • Arithmetic     →  computed instantly
      e.g.  "2 + 3 * (4 ^ 2)"

    • Knowledge      →  local KB first, then LLM
      e.g.  "What is quantum computing?"

    • Chat           →  pattern rules, then LLM
      e.g.  "Hello!"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print(text: str, color: str = "") -> None:
    print(f"{color}{text}{Style.RESET_ALL}" if color else text)


def _prompt() -> str:
    try:
        return input(f"\n{Fore.CYAN}You{Style.RESET_ALL} › ").strip()
    except (EOFError, KeyboardInterrupt):
        return "/exit"


def _respond(label: str, text: str) -> None:
    print(
        f"\n{Fore.GREEN}AnveshAI{Style.RESET_ALL} "
        f"[{Fore.YELLOW}{label}{Style.RESET_ALL}] › {text}"
    )


def _system(text: str) -> None:
    print(f"{Fore.MAGENTA}  {text}{Style.RESET_ALL}")


# ─────────────────────────────────────────────────────────────────────────────
# Response Composer
# ─────────────────────────────────────────────────────────────────────────────

def compose_response(
    user_input: str,
    intent: str,
    knowledge_engine: KnowledgeEngine,
    conversation_engine: ConversationEngine,
    llm_engine: LLMEngine,
    reasoning_engine: ReasoningEngine,
) -> tuple[str, str]:
    """
    Route input through the full hierarchy with chain-of-thought reasoning.
    Returns (label, response_text).

    Labels:
        Math             – plain arithmetic result (instant)
        AdvMath+CoT+LLM  – SymPy computed, reasoning planned, LLM explained
        AdvMath+CoT      – SymPy computed, reasoning-guided LLM fallback
        Knowledge        – local KB answered
        LLM+CoT-KB       – KB miss; reasoning-guided LLM answered
        Chat             – conversation pattern matched
        LLM+CoT          – reasoning-guided LLM answered
    """

    # ── Simple arithmetic ─────────────────────────────────────────────────────
    if intent == "math":
        return "Math", math_evaluate(user_input)

    # ── Advanced math ─────────────────────────────────────────────────────────
    if intent == "advanced_math":
        success, result_str, _latex = advanced_math_solve(user_input)

        if success:
            _system(f"SymPy → {result_str}")
            _system("Reasoning engine: decomposing problem…")
            plan = reasoning_engine.analyze(
                user_input, intent, has_symbolic_result=True
            )
            _system(plan.summary())
            if plan.warnings:
                for w in plan.warnings:
                    _system(f"  ⚠  {w}")
            _system("Building chain-of-thought prompt → LLM…")
            prompt = reasoning_engine.build_math_prompt(user_input, result_str, plan)
            explanation = llm_engine.generate(
                prompt,
                system_prompt=MATH_SYSTEM_PROMPT,
                temperature=MATH_TEMPERATURE,
            )
            full_response = (
                f"{result_str}\n\n"
                f"[Reasoning: {plan.problem_type} | {plan.strategy[:60]}]\n\n"
                f"{explanation}"
            )
            return "AdvMath+CoT+LLM", full_response

        else:
            _system(f"SymPy error: {result_str}")
            _system("Reasoning engine: building fallback chain-of-thought…")
            plan = reasoning_engine.analyze(user_input, intent)
            _system(plan.summary())
            prompt = reasoning_engine.build_math_fallback_prompt(
                user_input, plan, error_context=result_str
            )
            llm_response = llm_engine.generate(prompt)
            return "AdvMath+CoT", llm_response

    # ── Knowledge ─────────────────────────────────────────────────────────────
    if intent == "knowledge":
        kb_response, kb_found = knowledge_engine.query(user_input)
        if kb_found:
            return "Knowledge", kb_response

        _system("KB: no match — reasoning engine + LLM…")
        plan = reasoning_engine.analyze(user_input, intent)
        _system(plan.summary())
        prompt = reasoning_engine.build_general_prompt(
            user_input, intent, kb_response, plan
        )
        return "LLM+CoT-KB", llm_engine.generate(prompt)

    # ── Conversation ──────────────────────────────────────────────────────────
    chat_response, pattern_matched = conversation_engine.respond(user_input)
    if pattern_matched:
        return "Chat", chat_response

    _system("No pattern match — reasoning engine + LLM…")
    plan = reasoning_engine.analyze(user_input, intent)
    _system(plan.summary())
    prompt = reasoning_engine.build_general_prompt(user_input, intent, "", plan)
    return "LLM+CoT", llm_engine.generate(prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _print(BANNER, Fore.CYAN)
    _system("Initialising modules…")

    initialize_db()
    _system("✔  Memory (SQLite) ready")

    knowledge_engine    = KnowledgeEngine()
    _system("✔  Knowledge base loaded" if knowledge_engine.is_loaded() else "⚠  knowledge.txt not found")

    conversation_engine = ConversationEngine()
    _system("✔  Conversation engine ready")
    _system("✔  Math engine ready (AST safe-eval)")
    _system("✔  Advanced math engine ready (SymPy — 31+ operations)")
    _system("✔  Reasoning engine ready (chain-of-thought + CoT planning)")
    _system("✔  Intent router ready")

    llm_engine      = LLMEngine()
    reasoning_eng   = ReasoningEngine()
    _system("✔  LLM engine ready (Qwen2.5-0.5B loads on first use)")

    _print(f"\n{Fore.WHITE}Type /help for commands or just start chatting!{Style.RESET_ALL}")

    while True:
        user_input = _prompt()
        if not user_input:
            continue

        intent = classify_intent(user_input)

        # ── System commands ───────────────────────────────────────────────────
        if intent == "system":
            cmd = user_input.lower().split()[0]
            if cmd == "/exit":
                _print(f"\n{Fore.CYAN}Goodbye! Session closed.{Style.RESET_ALL}")
                sys.exit(0)
            elif cmd == "/history":
                _print(f"\n{Fore.YELLOW}── Conversation History ─────────────────────{Style.RESET_ALL}")
                _print(format_history())
                _print(f"{Fore.YELLOW}─────────────────────────────────────────────{Style.RESET_ALL}")
            elif cmd == "/help":
                _print(f"\n{Fore.YELLOW}── Help ──────────────────────────────────────{Style.RESET_ALL}")
                _print(HELP_TEXT)
                _print(f"{Fore.YELLOW}─────────────────────────────────────────────{Style.RESET_ALL}")
            else:
                _respond("System", f"Unknown command '{user_input}'. Type /help.")
            continue

        # ── Compose response ──────────────────────────────────────────────────
        label, response = compose_response(
            user_input, intent, knowledge_engine,
            conversation_engine, llm_engine, reasoning_eng
        )
        _respond(label, response)
        save_interaction(user_input, response)


if __name__ == "__main__":
    main()
