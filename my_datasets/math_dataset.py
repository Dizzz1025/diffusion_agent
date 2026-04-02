import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
    except Exception:
        pass
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{" in string:
        return string.split("\\text{")[0]
    return string


def _strip_string(string: str) -> str:
    if string is None:
        return ""
    string = str(string)
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    string = string.strip(".$")
    return string


def _extract_boxed_content(text: str) -> str:
    if not text:
        return ""
    matches = list(re.finditer(r"\\boxed", text))
    if not matches:
        return ""

    start = matches[-1].end()
    tail = text[start:].lstrip()
    if not tail:
        return ""

    if tail[0] == "{":
        depth = 0
        out = []
        for ch in tail[1:]:
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                if depth == 0:
                    break
                depth -= 1
                out.append(ch)
            else:
                out.append(ch)
        return "".join(out).strip()

    m = re.match(r"([^\s,$;]+)", tail)
    return m.group(1).strip() if m else ""


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    boxed = _extract_boxed_content(text)
    if boxed:
        return _strip_string(boxed)

    patterns = [
        r"[Tt]he answer is[:\s]*([^\n.]+)",
        r"[Ff]inal answer[:\s]*([^\n.]+)",
        r"[Aa]nswer[:\s]*([^\n.]+)",
    ]
    for pattern in patterns:
        ms = re.findall(pattern, text)
        if ms:
            return _strip_string(ms[-1])

    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^\$+|\$+$", "", last)
        return _strip_string(last)
    return ""


def _normalize_for_compare(ans: Any) -> str:
    s = _strip_string(str(ans) if ans is not None else "")
    s = s.replace(",", "")
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]
    return s


def _latex_frac_to_plain(s: str) -> str:
    prev = None
    cur = s
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while prev != cur:
        prev = cur
        cur = pattern.sub(r"(\1)/(\2)", cur)
    return cur


def _try_sympy_equiv(a: str, b: str) -> Optional[bool]:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transforms = standard_transformations + (implicit_multiplication_application,)
        aa = _latex_frac_to_plain(a).replace("^", "**")
        bb = _latex_frac_to_plain(b).replace("^", "**")
        expr_a = parse_expr(aa, transformations=transforms, evaluate=True)
        expr_b = parse_expr(bb, transformations=transforms, evaluate=True)
        return bool(simplify(expr_a - expr_b) == 0)
    except Exception:
        return None



def math_check_correctness(pred: Any, gt: Any) -> bool:
    pred_norm = _normalize_for_compare(pred)
    gt_norm = _normalize_for_compare(gt)
    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True

    sympy_eq = _try_sympy_equiv(pred_norm, gt_norm)
    if sympy_eq is not None:
        return sympy_eq

    return False



def math_get_predict(pred_str: Any) -> str:
    if isinstance(pred_str, (list, tuple)):
        pred_str = pred_str[0] if pred_str else ""
    pred = extract_final_answer(str(pred_str))
    return _normalize_for_compare(pred)



def load_math_records(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if path.is_dir():
        records = []
        for fp in sorted(path.rglob("*.json")):
            with fp.open("r", encoding="utf-8") as f:
                records.append(json.load(f))
        return records

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    raise ValueError(f"Unsupported MATH dataset path: {dataset_path}")



def math_data_process(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed = []
    for data in dataset:
        question = data.get("problem") or data.get("question") or data.get("task") or ""
        solution = data.get("solution") or data.get("detailed_solution") or data.get("rationale") or ""
        answer = data.get("answer") or data.get("final_answer") or extract_final_answer(solution)
        answer = _normalize_for_compare(answer)

        processed.append(
            {
                "task": question,
                "step": solution,
                "answer": answer,
                "level": data.get("level"),
                "subject": data.get("type") or data.get("subject") or data.get("category"),
                "raw": data,
            }
        )
    return processed
