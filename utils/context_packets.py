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


def build_output_packet(role: str, text: str, structured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    packet_type = infer_packet_type(role)

    structured = structured or {}

    packet = {
        "packet_type": packet_type,
        "role": role,
        "raw_text": text,
        "summary": structured.get("summary", "") or short_text(text, 300),
        "known_facts": structured.get("known_facts", []),
        "plan_steps": structured.get("plan_steps", []),
        "candidate_answer": structured.get("candidate_answer", "") or extract_final_answer(text),
        "verdict": structured.get("verdict", ""),
        "errors_found": structured.get("errors_found", []),
        "code_result": structured.get("code_result", ""),
        "final_text": structured.get("final_text", ""),
    }

    if packet_type == "program":
        packet["code"] = extract_code_block(text)

    if packet_type == "check" and not packet["verdict"]:
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


def select_context_packets(
    role: str,
    spatial_info: Dict[str, Dict[str, Any]],
    temporal_info: Dict[str, Dict[str, Any]],
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