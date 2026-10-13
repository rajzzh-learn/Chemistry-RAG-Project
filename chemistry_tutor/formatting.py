"""
Post-processing for LLM answers so they render cleanly in Streamlit.

Streamlit's markdown renderer does not support mhchem (`\\ce{...}`) and shows raw
HTML such as `<br>`. LLMs often emit both, so we convert them to plain Unicode
(e.g. `\\ce{CH3CHO + NH3 -> CH3CH=NH + H2O}` -> `CH₃CHO + NH₃ → CH₃CH=NH + H₂O`).
"""
from __future__ import annotations

import re

_SUB = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

_BR_RE = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)
_SUP_RE = re.compile(r"\^\{([^}]*)\}|\^([0-9]*[+-]?)")
_SUB_BRACED_RE = re.compile(r"_\{([^}]*)\}")
# digit(s) directly after a letter, ')' or ']' are subscripts (H2O, Ca(OH)2); coefficients follow a space/start/'+'
_SUBSCRIPT_DIGITS_RE = re.compile(r"(?<=[A-Za-z\)\]])(\d+)")
# charge written like Fe^{3+} is handled by _SUP_RE; run subscripts only on text outside charges
_ARROWS = (("<=>", "⇌"), ("<->", "⇌"), ("<-", "←"), ("->", "→"), ("\\rightarrow", "→"), ("\\to", "→"))


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Return index of the '}' matching the '{' at open_idx, or -1."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _ce_to_unicode(formula: str) -> str:
    for src, dst in _ARROWS:
        formula = formula.replace(src, dst)
    formula = _SUB_BRACED_RE.sub(lambda m: m.group(1).translate(_SUB), formula)

    # Protect superscript charges, then subscript the remaining digits.
    sups: list[str] = []

    def _stash_sup(m: re.Match) -> str:
        sups.append((m.group(1) if m.group(1) is not None else m.group(2)).translate(_SUP))
        return f"\x00{len(sups) - 1}\x00"

    formula = _SUP_RE.sub(_stash_sup, formula)
    formula = _SUBSCRIPT_DIGITS_RE.sub(lambda m: m.group(1).translate(_SUB), formula)
    formula = re.sub(r"\x00(\d+)\x00", lambda m: sups[int(m.group(1))], formula)
    return formula.replace("\\,", " ").replace("{", "").replace("}", "")


def _convert_ce(text: str) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("\\ce{", pos)
        if start == -1:
            out.append(text[pos:])
            break
        end = _find_matching_brace(text, start + 3)
        if end == -1:
            out.append(text[pos:])
            break
        before, after = text[pos:start], text[end + 1:]
        converted = _ce_to_unicode(text[start + 4:end])
        # Drop wrappers the model added around the formula: $...$ or (...)
        for wrap_open, wrap_close in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("(", ")")):
            if before.endswith(wrap_open) and after.startswith(wrap_close):
                before = before[: -len(wrap_open)]
                after = after[len(wrap_close):]
                break
        out.append(before)
        out.append(converted)
        text = after
        pos = 0
    return "".join(out)


def clean_answer(text: str) -> str:
    """Convert mhchem and HTML line breaks in an LLM answer to Streamlit-friendly markdown."""
    if not text:
        return text
    text = _BR_RE.sub("\n\n", text)
    if "\\ce{" in text:
        text = _convert_ce(text)
    return text
