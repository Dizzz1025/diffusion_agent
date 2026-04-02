# utils/context_packets.py
import re
from typing import Any, Dict, List, Optional


ROLE_PACKET_POLICY = {
    "ProblemDecomposer": {
        "spatial_types": [],
        "temporal_types": [],
    },
    "MathSolver": {
        "spatial_types": ["plan", "program", "check", "solve"],
        "temporal_types": ["plan", "program"],
    },
    "ProgrammingExpert": {
        "spatial_types": ["plan", "solve"],
        "temporal_types": ["plan", "solve"],
    },
    "CalculationChecker": {
        "spatial_types": ["solve", "program", "plan"],
        "temporal_types": ["solve", "program"],
    },
}

ROLE_PACKET_VIEW = {
    "ProblemDecomposer": {
        "spatial_fields": ["summary", "known_facts", "plan_steps"],
        "temporal_fields": ["summary", "plan_steps"],
        "max_items": 3,
    },
    "MathSolver": {
        "spatial_fields": ["summary", "known_facts", "plan_steps", "candidate_answer", "verdict", "code_result", "errors_found"],
        "temporal_fields": ["summary", "plan_steps", "candidate_answer", "code_result"],
        "max_items": 4,
    },
    "ProgrammingExpert": {
        "spatial_fields": ["summary", "known_facts", "plan_steps", "candidate_answer"],
        "temporal_fields": ["summary", "plan_steps", "candidate_answer"],
        "max_items": 4,
    },
    "CalculationChecker": {
        "spatial_fields": ["summary", "known_facts", "plan_steps", "candidate_answer", "code_result"],
        "temporal_fields": ["summary", "candidate_answer", "code_result"],
        "max_items": 4,
    },
    "default": {
        "spatial_fields": ["summary", "known_facts", "plan_steps", "candidate_answer"],
        "temporal_fields": ["summary", "candidate_answer"],
        "max_items": 4,
    },
}

ROLE_PACKET_SCHEMA = {
    "ProblemDecomposer": {
        "public_fields": ["summary", "known_facts", "plan_steps", "final_text"],
        "extra_fields": ["variables", "constraints", "equations"],
    },
    "MathSolver": {
        "public_fields": ["summary", "known_facts", "plan_steps", "candidate_answer", "final_text"],
        "extra_fields": ["used_packets", "derivation", "sanity_check", "low_confidence"],
    },
    "ProgrammingExpert": {
        "public_fields": ["summary", "known_facts", "candidate_answer", "code_result", "final_text"],
        "extra_fields": ["formulation", "python_code"],
    },
    "CalculationChecker": {
        "public_fields": ["summary", "known_facts", "plan_steps", "candidate_answer", "verdict", "errors_found", "final_text"],
        "extra_fields": ["target_answer", "recomputation", "correct_answer"],
    },
    "default": {
        "public_fields": ["summary", "known_facts", "plan_steps", "candidate_answer", "final_text"],
        "extra_fields": [],
    },
}

def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"####\s*Answer:\s*([^\n]+)",
        r"[Tt]he answer is\s*([^\n\.]+)",
        r"[Ff]inal answer[:：]\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return ""


def extract_code_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```python\s*(.*?)```", text, flags=re.S)
    return m.group(1).strip() if m else ""


def short_text(text: str, max_len: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


def infer_packet_type(role: str) -> str:
    mapping = {
        "ProblemDecomposer": "plan",
        "MathSolver": "solve",
        "ProgrammingExpert": "program",
        "CalculationChecker": "check",
    }
    return mapping.get(role, "generic")


def build_output_packet(
    role: str,
    text: str,
    structured: Optional[Dict[str, Any]] = None,
    selected_packets: Optional[List[str]] = None,
    output_packet_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    packet_type = infer_packet_type(role)
    structured = structured or {}
    selected_packets = selected_packets or []
    output_packet_types = output_packet_types or []

    schema = ROLE_PACKET_SCHEMA.get(role, ROLE_PACKET_SCHEMA["default"])

    packet = {
        "packet_type": packet_type,
        "role": role,
        "raw_text": text,
    }

    # 1) 先写公共字段
    for field in schema["public_fields"]:
        if field == "summary":
            packet[field] = structured.get("summary", "") or short_text(text, 300)
        elif field == "candidate_answer":
            packet[field] = structured.get("candidate_answer", "") or extract_final_answer(text)
        else:
            value = structured.get(field)
            if value not in (None, "", [], {}):
                packet[field] = value

    # 2) 再写角色专属字段
    for field in schema.get("extra_fields", []):
        value = structured.get(field)
        if value not in (None, "", [], {}):
            packet[field] = value

    # 3) 新增：记录本轮用了哪些上游 packet 类型
    if selected_packets:
        packet["selected_packets"] = list(dict.fromkeys([str(x) for x in selected_packets if x]))

    # 4) 新增：记录本轮产出了哪些输出证据类型
    if output_packet_types:
        packet["output_packet_types"] = list(dict.fromkeys([str(x) for x in output_packet_types if x]))

    # 5) 兼容旧逻辑
    if packet_type == "program" and not packet.get("code"):
        code = extract_code_block(text)
        if code:
            packet["code"] = code

    if packet_type == "check" and not packet.get("verdict"):
        lowered = (text or "").lower()
        packet["verdict"] = (
            "incorrect" if "incorrect" in lowered or "error" in lowered
            else "correct" if "correct" in lowered
            else "unknown"
        )

    return packet


def _match_packet_types(packet: Optional[Dict[str, Any]], allowed: List[str]) -> bool:
    if packet is None:
        return False
    if not allowed:
        return False
    return packet.get("packet_type") in allowed

def _rerank_packets_with_prior(
    selected_items: List[Dict[str, Any]],
    role: str,
    execution_memory=None,
) -> List[Dict[str, Any]]:
    execution_memory = execution_memory or {}
    role_packet_priors = execution_memory.get("role_packet_priors", {}) or {}
    pkt_prior = role_packet_priors.get(role, {}) or {}

    def _packet_score(item):
        packet = item.get("packet", {}) or {}
        pkt_type = str(packet.get("packet_type", ""))

        # 如果 packet 本身带有 score，就用它；没有就给一个默认值
        base = float(packet.get("score", 1.0)) if isinstance(packet, dict) else 1.0

        # memory 里记录的是 “这个 role 更偏好哪些 packet_type”
        prior = float(pkt_prior.get(pkt_type, 0.0))

        return base + 0.75 * prior

    return sorted(selected_items, key=_packet_score, reverse=True)

def select_context_packets(
    role: str,
    spatial_info: Dict[str, Dict[str, Any]],
    temporal_info: Dict[str, Dict[str, Any]],
    execution_memory=None
) -> Dict[str, List[Dict[str, Any]]]:
    policy = ROLE_PACKET_POLICY.get(
        role,
        {"spatial_types": ["plan", "solve", "program", "check"], "temporal_types": []},
    )

    selected_spatial = []
    for node_id, info in spatial_info.items():
        packet = info.get("packet")
        if _match_packet_types(packet, policy["spatial_types"]):
            selected_spatial.append({
                "node_id": node_id,
                "role": info.get("role", ""),
                "packet": packet,
            })

    selected_temporal = []
    for node_id, info in temporal_info.items():
        packet = info.get("packet")
        if _match_packet_types(packet, policy["temporal_types"]):
            selected_temporal.append({
                "node_id": node_id,
                "role": info.get("role", ""),
                "packet": packet,
            })

    # 关键新增：规则筛完以后，再按 memory prior 软重排
    selected_spatial = _rerank_packets_with_prior(
        selected_spatial,
        role=role,
        execution_memory=execution_memory,
    )
    selected_temporal = _rerank_packets_with_prior(
        selected_temporal,
        role=role,
        execution_memory=execution_memory,
    )

    return {
        "spatial": selected_spatial,
        "temporal": selected_temporal,
    }

def build_memory_packets(memory_summary: Optional[Dict[str, Any]], top_k: int = 2) -> List[Dict[str, Any]]:
    if not memory_summary:
        return []

    retrieved = memory_summary.get("retrieved", []) or []
    packets = []

    for item in retrieved[:top_k]:
        packets.append({
            "score": item.get("score", 0.0),
            "trace": item.get("trace", item.get("selected_trace", [])),
            "correct": item.get("correct", 0),
            "reward": item.get("reward", 0.0),
            "task_text": short_text(item.get("task_text", ""), 120),
        })

    return packets


def format_packet_context(selected: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = []

    def render_packet(item):
        pkt = item["packet"]
        out = [f"- from {item['node_id']} ({item['role']}, {pkt['packet_type']}):"]

        if pkt.get("summary"):
            out.append(f"  summary: {pkt['summary']}")

        if pkt.get("known_facts"):
            out.append("  known_facts:")
            for fact in pkt["known_facts"][:4]:
                out.append(f"  - {fact}")

        if pkt.get("plan_steps"):
            out.append("  plan_steps:")
            for step in pkt["plan_steps"][:3]:
                out.append(f"  - {step}")

        if pkt.get("candidate_answer"):
            out.append(f"  candidate_answer: {pkt['candidate_answer']}")

        if pkt.get("verdict"):
            out.append(f"  verdict: {pkt['verdict']}")

        if pkt.get("errors_found"):
            out.append("  errors_found:")
            for err in pkt["errors_found"][:3]:
                out.append(f"  - {err}")

        if pkt.get("code_result"):
            out.append(f"  code_result: {pkt['code_result']}")

        if pkt.get("final_text"):
            out.append(f"  final_text: {pkt['final_text']}")

        return "\n".join(out)

    if selected["spatial"]:
        lines.append("[Visible collaboration packets]")
        for item in selected["spatial"]:
            lines.append(render_packet(item))

    if selected["temporal"]:
        lines.append("\n[Temporal packets]")
        for item in selected["temporal"]:
            lines.append(render_packet(item))

    return "\n".join(lines).strip()

def _append_field_lines(lines: List[str], pkt: Dict[str, Any], fields: List[str], indent: str = "  ") -> None:
    for field in fields:
        value = pkt.get(field)

        if not value:
            continue

        if isinstance(value, list):
            if len(value) == 0:
                continue
            lines.append(f"{indent}{field}:")
            for item in value[:4]:
                lines.append(f"{indent}- {short_text(str(item), 160)}")
        else:
            lines.append(f"{indent}{field}: {short_text(str(value), 220)}")


def render_role_specific_packet_context(role: str, selected: Dict[str, List[Dict[str, Any]]]) -> str:
    view = ROLE_PACKET_VIEW.get(role, ROLE_PACKET_VIEW["default"])
    spatial_fields = view["spatial_fields"]
    temporal_fields = view["temporal_fields"]
    max_items = view.get("max_items", 4)

    lines = []

    spatial_packets = selected.get("spatial", [])[:max_items]
    temporal_packets = selected.get("temporal", [])[:max_items]

    if spatial_packets:
        lines.append("[Spatial packets]")
        for item in spatial_packets:
            pkt = item.get("packet", {}) or {}
            lines.append(
                f"- from {item.get('node_id', '?')} "
                f"({item.get('role', 'Unknown')}, {pkt.get('packet_type', 'generic')}):"
            )
            _append_field_lines(lines, pkt, spatial_fields)

    if temporal_packets:
        if lines:
            lines.append("")
        lines.append("[Temporal packets]")
        for item in temporal_packets:
            pkt = item.get("packet", {}) or {}
            lines.append(
                f"- from {item.get('node_id', '?')} "
                f"({item.get('role', 'Unknown')}, {pkt.get('packet_type', 'generic')}):"
            )
            _append_field_lines(lines, pkt, temporal_fields)

    return "\n".join(lines).strip()

def format_memory_context(memory_packets: List[Dict[str, Any]]) -> str:
    if not memory_packets:
        return ""

    lines = ["[Relevant memory]"]
    for i, item in enumerate(memory_packets, 1):
        lines.append(
            f"- memory#{i}: trace={item['trace']}, correct={item['correct']}, "
            f"reward={item['reward']:.3f}, task={item['task_text']}"
        )
    return "\n".join(lines)

def _extract_answer_hint(self, text: str) -> str:
    patterns = [
        r"####\s*Answer:\s*([^\n]+)",
        r"[Tt]he answer is\s*([^\n\.]+)",
        r"[Ff]inal answer[:：]\s*([^\n]+)",
        r"[Cc]andidate[_ ]?[Aa]nswer[:：]?\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return ""

def _top_items(d: Dict[str, Any], k: int = 3):
    if not isinstance(d, dict):
        return []
    items = []
    for key, val in d.items():
        try:
            score = float(val)
        except Exception:
            continue
        items.append((key, score))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:k]


def build_execution_memory_block(role: str, execution_memory: Dict[str, Any]) -> str:
    if not execution_memory:
        return ""

    role_hints = (execution_memory.get("role_hints", {}) or {}).get(role, {}) or {}
    packet_priors = (execution_memory.get("role_packet_priors", {}) or {}).get(role, {}) or {}
    output_priors = (execution_memory.get("role_output_packet_priors", {}) or {}).get(role, {}) or {}

    times_seen = int(role_hints.get("times_seen", 0) or 0)
    if times_seen < 3:
        return ""

    lines = [f"[Execution memory for {role}]"]

    avg_position = role_hints.get("avg_position", None)
    if avg_position is not None:
        try:
            lines.append(
                f"In similar successful traces, this role usually appears around step {float(avg_position):.1f}."
            )
        except Exception:
            pass

    prev_roles = [name for name, _ in _top_items(role_hints.get("common_prev_roles", {}), k=2)]
    next_roles = [name for name, _ in _top_items(role_hints.get("common_next_roles", {}), k=2)]
    input_types = [name for name, _ in _top_items(packet_priors, k=3)]
    output_types = [name for name, _ in _top_items(output_priors, k=3)]

    if prev_roles:
        lines.append("This role usually follows: " + ", ".join(prev_roles) + ".")
    if next_roles:
        lines.append("Its outputs are often passed to: " + ", ".join(next_roles) + ".")
    if input_types:
        lines.append("Prioritize input packet types: " + ", ".join(input_types) + ".")
    if output_types:
        lines.append("Typical output packet types: " + ", ".join(output_types) + ".")

    lines.append("Use these as soft hints only; follow the current task and visible context first.")

    return "\n".join(lines)