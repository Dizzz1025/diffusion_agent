from typing import List, Any, Dict, Optional

from ..graph.node import Node
from .agent_registry import AgentRegistry
from ..llm.llm_registry import LLMRegistry
from ..prompt.prompt_set_registry import PromptSetRegistry
from ..tools.coding.python_executor import execute_code_get_return
from my_datasets.gsm8k_dataset import gsm_get_predict
from utils.context_packets import (
    build_output_packet,
    build_memory_packets,
    select_context_packets,
    format_packet_context,
    render_role_specific_packet_context,
    format_memory_context,
    build_execution_memory_block,
)
import re
from textwrap import dedent

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "", **kwargs):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
   
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        system_prompt = self.constraint
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"], role=self.role)
        memory_summary = kwargs.get("memory_summary", {}) or {}

        selected_packets = select_context_packets(
            role=self.role,
            spatial_info=spatial_info,
            temporal_info=temporal_info,
            execution_memory=memory_summary.get("execution_memory", {}) if isinstance(memory_summary, dict) else {},
        )

        targeted_context_blocks: List[str] = []

        # Prefer one clean target over a long mixed history.
        if self.role == "CalculationChecker":
            latest_solver_packet = self._find_latest_packet_by_roles(
                selected_packets, ["MathSolver", "ProgrammingExpert"]
            )
            if latest_solver_packet:
                targeted_context_blocks.append(
                    self._render_checker_target_context(latest_solver_packet)
                )
        elif self.role == "MathSolver":
            latest_checker_packet = self._find_latest_packet_by_roles(
                selected_packets, ["CalculationChecker"]
            )
            if latest_checker_packet:
                targeted_context_blocks.append(
                    self._render_solver_revision_context(latest_checker_packet)
                )

        packet_context = render_role_specific_packet_context(self.role, selected_packets)
        if self.role in {"MathSolver", "CalculationChecker"} and targeted_context_blocks:
            # Avoid re-introducing stale historical candidates after we already selected the latest target.
            pass
        elif packet_context:
            targeted_context_blocks.append(
                f"[Structured predecessor context]\n{packet_context}"
            )
        else:
            targeted_context_blocks.append("[Visible collaboration packets]\n- None")

        execution_memory_block = build_execution_memory_block(
            self.role,
            memory_summary.get("execution_memory", {}) if isinstance(memory_summary, dict) else {},
        )
        if execution_memory_block:
            targeted_context_blocks.append(execution_memory_block)

        user_prompt += "\n\n" + "\n\n".join(block for block in targeted_context_blocks if block)
        user_prompt += "\n\n" + self._get_role_task_block()
        user_prompt += "\n\n" + self._get_role_output_format_block()

        self._selected_packet_keys = self._collect_selected_packet_types(selected_packets)
        return system_prompt, user_prompt
    
    def _get_role_task_block(self) -> str:
        role_task_map = {
            "ProblemDecomposer": (
                "[Your task]\n"
                "Analyze the problem only. Extract variables, quantities, constraints, and a step plan.\n"
                "Do NOT compute the final answer.\n"
                "Do not guess missing information.\n"
                "Highlight the critical geometric/algebraic quantity that must be computed correctly.\n"
                "If a naive symmetric shortcut may be invalid, mention that the correct quantity should be derived carefully.\n"
                "Focus on problem decomposition rather than long free-form reasoning."
            ),
            "MathSolver": (
                "[Your task]\n"
                "Solve the problem step by step.\n"
                "If checker feedback is provided, revise the previous solution by directly fixing the flagged issue.\n"
                "Do not ignore checker feedback.\n"
                "Do not jump to an unsupported answer.\n"
                "End with '####Answer: <final_answer>'."
            ),
            "ProgrammingExpert": (
                "[Your task]\n"
                "Translate the math problem into executable Python when helpful.\n"
                "Focus on formulas, expressions, and program result.\n"
                "Write runnable Python code only when code is actually useful.\n"
                "The code must assign the final result to a variable named answer.\n"
                "The last meaningful line of code should be in the form: answer = <final_result>.\n"
                "Do not mix explanation inside the code block."
            ),
            "CalculationChecker": (
                "[Your task]\n"
                "Check only the latest candidate solution provided in the targeted context.\n"
                "Your main job is to identify errors, missing steps, unsupported jumps, or invalid formulas.\n"
                "Do not drift to older candidate answers unless they are explicitly marked as the current target.\n"
                "You do not need to fully solve the problem unless necessary.\n"
                "If the solution is flawed, provide actionable revision feedback for MathSolver.\n"
                "Output a clear verdict using exactly one of: correct / incorrect / uncertain."
            ),
        }
        return role_task_map.get(
            self.role,
            "[Your task]\nSolve the problem carefully."
        )

    def _get_role_output_format_block(self) -> str:
        role_format_map = {
            "ProblemDecomposer": dedent("""
                [Output format]
                Use exactly these sections.

                [SUMMARY]
                A brief description of the problem structure.

                [VARIABLES]
                List the key variables / quantities.
                Use bullet points starting with "-".

                [CONSTRAINTS]
                List the constraints / conditions.
                Use bullet points starting with "-".

                [PLAN]
                List the intended solving steps only.
                Use bullet points starting with "-".
                Do NOT solve them out.

                [EQUATIONS]
                List useful equations or relations if available.

                [FINAL]
                One-sentence decomposition conclusion only.
            """).strip(),

            "MathSolver": dedent("""
                [Output format]
                Use exactly these sections.

                [SUMMARY]
                A short summary of the solution idea.

                [USED_PACKETS]
                List which upstream packets were useful.
                Use bullet points starting with "-".

                [DERIVATION]
                Show the essential derivation steps in order.
                Use bullet points starting with "-".
                Every nontrivial formula must be connected to the current problem.
                Do not skip the step that directly leads to the final answer.

                [SANITY_CHECK]
                Briefly check whether the result is reasonable.
                Examples: sign, magnitude, substitution, geometric feasibility.
                If no simple check is available, write "None".

                [CANDIDATE_ANSWER]
                Write the final candidate answer only.

                [FINAL]
                State the final conclusion briefly, and end with:
                ####Answer: <final_answer>
            """).strip(),

            "ProgrammingExpert": dedent("""
                [Output format]
                Use exactly these sections.

                [SUMMARY]
                A short summary of the computational approach.

                [FORMULATION]
                Briefly describe how the problem is converted into formulas / code.

                [PYTHON_CODE]
                Provide only Python code in a fenced code block.
                The code must assign the final result to a variable named answer.

                [CODE_RESULT]
                Report the value of answer.

                [CANDIDATE_ANSWER]
                Write the candidate numerical answer.

                [FINAL]
                State the final conclusion briefly.
            """).strip(),

            "CalculationChecker": dedent("""
                [Output format]
                Use exactly these sections.

                [SUMMARY]
                A short summary of what was checked.

                [TARGET_ANSWER]
                State the answer / computation target being checked.

                [RECOMPUTATION]
                Recompute the critical quantities needed to verify the answer.
                Use bullet points starting with "-".
                If the original derivation is incomplete, explicitly fill in the missing steps.

                [VERDICT]
                Output one of: correct / incorrect / uncertain

                [ERRORS]
                List specific errors if found.
                Use bullet points starting with "-".
                If none, write "None".

                [CORRECT_ANSWER]
                If the target answer is wrong and you can determine the right one, write it here.

                [FINAL]
                State the final checking conclusion briefly.
            """).strip(),
        }

        return role_format_map.get(
            self.role,
            dedent("""
                [Output format]

                [SUMMARY]
                ...

                [FINAL]
                ...
            """).strip()
        )

    def _collect_selected_packet_types(self, selected_packets: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        packet_types = []
        for bucket in ("spatial", "temporal"):
            for item in selected_packets.get(bucket, []):
                pkt = item.get("packet", {}) or {}
                pkt_type = str(pkt.get("packet_type", "")).strip()
                if pkt_type:
                    packet_types.append(pkt_type)
        # 去重但保序
        return list(dict.fromkeys(packet_types))

    def _infer_output_packet_types(self, role: str, structured: Dict[str, Any]) -> List[str]:
        types = []

        if structured.get("summary"):
            types.append("summary")

        if role == "ProblemDecomposer":
            if structured.get("plan_steps"):
                types.append("plan_steps")
            if structured.get("known_facts"):
                types.append("known_facts")
            if structured.get("final_text"):
                types.append("final")

        elif role == "MathSolver":
            if structured.get("derivation") or structured.get("plan_steps"):
                types.append("key_steps")
            if structured.get("candidate_answer"):
                types.append("candidate_answer")
            if structured.get("sanity_check"):
                types.append("sanity_check")
            if structured.get("final_text"):
                types.append("final")

        elif role == "ProgrammingExpert":
            if structured.get("python_code"):
                types.append("python_code")
            if structured.get("code_result"):
                types.append("code_result")
            if structured.get("candidate_answer"):
                types.append("candidate_answer")
            if structured.get("final_text"):
                types.append("final")

        elif role == "CalculationChecker":
            if structured.get("verdict"):
                types.append("verdict")
            if structured.get("errors_found"):
                types.append("errors_found")
            if structured.get("checked_answer"):
                types.append("checked_answer")
            if structured.get("corrected_answer"):
                types.append("corrected_answer")
            if structured.get("final_text"):
                types.append("final")

        return types

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)

        executed_code_result = ""
        if self.role == "ProgrammingExpert":
            code = self._extract_python_code(response)
            if code:
                try:
                    executed_code_result = str(execute_code_get_return(code))
                except Exception as e:
                    executed_code_result = f"EXECUTION_ERROR: {e}"

        structured = self._extract_structured_packet(response, self.role)
        if self.role == "ProgrammingExpert" and executed_code_result:
            structured["code_result"] = executed_code_result
            if not structured.get("candidate_answer"):
                structured["candidate_answer"] = executed_code_result

        self.output_packet = build_output_packet(
            self.role,
            response,
            structured=structured,
            selected_packets=getattr(self, "_selected_packet_keys", []),
            output_packet_types=self._infer_output_packet_types(self.role, structured),
        )
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)

        executed_code_result = ""
        if self.role == "ProgrammingExpert":
            code = self._extract_python_code(response)
            if code:
                try:
                    executed_code_result = str(execute_code_get_return(code))
                except Exception as e:
                    executed_code_result = f"EXECUTION_ERROR: {e}"

        structured = self._extract_structured_packet(response, self.role)
        if self.role == "ProgrammingExpert" and executed_code_result:
            structured["code_result"] = executed_code_result
            if not structured.get("candidate_answer"):
                structured["candidate_answer"] = executed_code_result

        self.output_packet = build_output_packet(
            self.role,
            response,
            structured=structured,
            selected_packets=getattr(self, "_selected_packet_keys", []),
            output_packet_types=self._infer_output_packet_types(self.role, structured),
        )
        if kwargs.get("debug", False):
            print(f"#################system_prompt:{system_prompt}")
            print(f"#################user_prompt:{user_prompt}")
            print(f"#################response:{response}")
        return response
    
    def _extract_section(self, text: str, section_name: str) -> str:
        sec = re.escape(section_name)
        current_header = rf"(?:\[\s*{sec}\s*\]|\*\*\s*\[\s*{sec}\s*\]\s*\*\*)"
        any_header = r"(?:\[\s*[A-Z_ ]+\s*\]|\*\*\s*\[\s*[A-Z_ ]+\s*\]\s*\*\*)"
        pattern = rf"^\s*{current_header}\s*:?\s*(.*?)(?=^\s*{any_header}\s*:?\s*|\Z)"
        m = re.search(pattern, text, flags=re.S | re.M | re.I)
        return m.group(1).strip() if m else ""
    
    def _extract_bullets(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                lines.append(line[1:].strip())
            elif line.startswith("*"):
                lines.append(line[1:].strip())
            elif re.match(r"^\d+\.", line):
                lines.append(re.sub(r"^\d+\.\s*", "", line))
        return lines
    
    def _extract_bullets_from_first_available(self, text: str, section_names: List[str]) -> List[str]:
        for name in section_names:
            sec = self._extract_section(text, name)
            if sec:
                return self._extract_bullets(sec)
        return []

    def _count_substantive_steps(self, steps: List[str]) -> int:
        cnt = 0
        for s in steps:
            s = s.strip()
            if len(s) >= 12:
                cnt += 1
        return cnt


    def _flatten_selected_packets(self, selected_packets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        flat: List[Dict[str, Any]] = []
        for bucket in ("spatial", "temporal"):
            for item in selected_packets.get(bucket, []):
                packet = (item or {}).get("packet", {}) or {}
                if packet:
                    flat.append(packet)
        return flat

    def _find_latest_packet_by_roles(
        self,
        selected_packets: Dict[str, List[Dict[str, Any]]],
        roles: List[str],
    ) -> Optional[Dict[str, Any]]:
        role_set = {str(r).strip() for r in roles}
        flat = self._flatten_selected_packets(selected_packets)
        for packet in reversed(flat):
            if str(packet.get("role", "")).strip() in role_set:
                structured = packet.get("structured", {}) or {}
                if structured:
                    return packet
        return None

    def _normalize_verdict(self, verdict: str) -> str:
        verdict = (verdict or "").strip().lower()
        mapping = {
            "pass": "correct",
            "correct": "correct",
            "ok": "correct",
            "revise": "incorrect",
            "fail": "incorrect",
            "incorrect": "incorrect",
            "wrong": "incorrect",
            "uncertain": "uncertain",
            "unknown": "uncertain",
        }
        return mapping.get(verdict, verdict)

    def _render_checker_target_context(self, packet: Dict[str, Any]) -> str:
        structured = packet.get("structured", {}) or {}
        role = str(packet.get("role", "")).strip() or "UnknownRole"
        candidate_answer = (
            structured.get("candidate_answer")
            or structured.get("corrected_answer")
            or structured.get("checked_answer")
            or ""
        )
        derivation = structured.get("derivation") or structured.get("plan_steps") or []
        if isinstance(derivation, str):
            derivation = [derivation]
        derivation = [str(x).strip() for x in derivation if str(x).strip()]

        lines = [
            "[Target solution to check]",
            f"Source role: {role}",
            f"Current candidate answer: {candidate_answer or 'None'}",
        ]
        if derivation:
            lines.append("Current derivation / key steps:")
            lines.extend([f"- {step}" for step in derivation[:6]])
        final_text = structured.get("final_text", "")
        if final_text:
            lines.append(f"Final statement: {final_text}")
        lines.append("Only check this current target. Do not drift to earlier candidates unless explicitly referenced here.")
        return "\n".join(lines)

    def _render_solver_revision_context(self, packet: Dict[str, Any]) -> str:
        structured = packet.get("structured", {}) or {}
        verdict = self._normalize_verdict(structured.get("verdict", ""))
        checked_answer = structured.get("checked_answer") or structured.get("target_answer") or structured.get("candidate_answer") or ""
        corrected_answer = structured.get("corrected_answer") or structured.get("correct_answer") or ""
        errors_found = structured.get("errors_found") or []
        if isinstance(errors_found, str):
            errors_found = [errors_found]
        errors_found = [str(x).strip() for x in errors_found if str(x).strip()]

        lines = [
            "[Latest checker feedback]",
            f"Verdict: {verdict or 'uncertain'}",
            f"Checked answer: {checked_answer or 'None'}",
        ]
        if errors_found:
            lines.append("Errors found:")
            lines.extend([f"- {err}" for err in errors_found[:6]])
        else:
            lines.append("Errors found:")
            lines.append("- None")
        if corrected_answer:
            lines.append(f"Suggested corrected answer: {corrected_answer}")
        lines.append("Revise the previous solution by directly fixing the flagged issue. Do not restart from unrelated older candidates.")
        return "\n".join(lines)

    def _extract_structured_packet(self, text: str, role: str) -> Dict[str, Any]:
        packet = {
            "role": role,
            "summary": "",
            "known_facts": [],
            "plan_steps": [],
            "candidate_answer": "",
            "verdict": "",
            "errors_found": [],
            "code_result": "",
            "final_text": "",
        }

        if role == "ProblemDecomposer":
            summary = self._extract_section(text, "SUMMARY")
            variables = self._extract_bullets(self._extract_section(text, "VARIABLES"))
            constraints = self._extract_bullets(self._extract_section(text, "CONSTRAINTS"))
            plan_steps = self._extract_bullets(self._extract_section(text, "PLAN"))
            equations = self._extract_lines(self._extract_section(text, "EQUATIONS"))
            final_text = self._extract_section(text, "FINAL")

            packet.update({
                "summary": summary or self._truncate(text, 240),
                "known_facts": variables + constraints,
                "plan_steps": plan_steps,
                "final_text": final_text.strip() if final_text else "",
                "variables": variables,
                "constraints": constraints,
                "equations": equations,
            })
            return packet

        if role == "MathSolver":
            summary = self._extract_section(text, "SUMMARY")
            used_packets = self._extract_bullets(self._extract_section(text, "USED_PACKETS"))
            derivation = self._extract_bullets_from_first_available(text, ["DERIVATION", "KEY_STEPS"])
            sanity_check = self._extract_section(text, "SANITY_CHECK")
            candidate_answer = self._extract_section(text, "CANDIDATE_ANSWER")
            final_text = self._extract_section(text, "FINAL")

            if not candidate_answer:
                candidate_answer = self._extract_answer_hint(text)

            low_confidence = bool(candidate_answer.strip()) and self._count_substantive_steps(derivation) < 2

            packet.update({
                "summary": summary or self._truncate(text, 240),
                "known_facts": [],
                "plan_steps": derivation,              # 兼容旧下游
                "candidate_answer": candidate_answer.strip() if candidate_answer else "",
                "final_text": final_text.strip() if final_text else "",
                "used_packets": used_packets,
                "derivation": derivation,
                "sanity_check": sanity_check.strip() if sanity_check else "",
                "low_confidence": low_confidence,
            })
            return packet

        if role == "ProgrammingExpert":
            summary = self._extract_section(text, "SUMMARY")
            formulation = self._extract_section(text, "FORMULATION")
            code = self._extract_python_code(text)
            code_result = self._extract_section(text, "CODE_RESULT")
            candidate_answer = self._extract_section(text, "CANDIDATE_ANSWER")
            final_text = self._extract_section(text, "FINAL")

            if not candidate_answer:
                candidate_answer = self._extract_answer_hint(text)

            packet.update({
                "summary": summary or self._truncate(text, 240),
                "known_facts": self._extract_lines(formulation),
                "plan_steps": [],
                "candidate_answer": candidate_answer.strip() if candidate_answer else "",
                "code_result": code_result.strip() if code_result else "",
                "final_text": final_text.strip() if final_text else "",
                "formulation": formulation.strip() if formulation else "",
                "python_code": code,
            })
            return packet

        if role == "CalculationChecker":
            summary = self._extract_section(text, "SUMMARY")
            target_answer = self._extract_section(text, "TARGET_ANSWER")
            recomputation = self._extract_bullets_from_first_available(text, ["RECOMPUTATION", "CHECK_STEPS"])
            verdict = self._normalize_verdict(self._extract_section(text, "VERDICT"))
            errors = self._extract_bullets(self._extract_section(text, "ERRORS"))
            correct_answer = self._extract_section(text, "CORRECT_ANSWER")
            final_text = self._extract_section(text, "FINAL")

            if not correct_answer and verdict == "correct":
                correct_answer = target_answer

            packet.update({
                "summary": summary or self._truncate(text, 240),
                "known_facts": [target_answer] if target_answer else [],
                "plan_steps": recomputation,           # 兼容旧下游
                "candidate_answer": target_answer.strip() if target_answer else "",
                "checked_answer": target_answer.strip() if target_answer else "",
                "corrected_answer": correct_answer.strip() if correct_answer else "",
                "verdict": verdict if verdict else "",
                "errors_found": errors,
                "final_text": final_text.strip() if final_text else "",
                "target_answer": target_answer.strip() if target_answer else "",
                "correct_answer": correct_answer.strip() if correct_answer else "",
                "recomputation": recomputation,
            })
            return packet

        # fallback
        summary = self._extract_section(text, "SUMMARY")
        final_text = self._extract_section(text, "FINAL")
        candidate_answer = self._extract_answer_hint(text)

        packet.update({
            "summary": summary or self._truncate(text, 240),
            "candidate_answer": candidate_answer.strip() if candidate_answer else "",
            "final_text": final_text.strip() if final_text else "",
        })
        return packet
    
    def _extract_lines(self, text: str) -> List[str]:
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*]\s*", "", line)
            items.append(line)
        return items

    def _extract_python_code(self, text: str) -> str:
        # 优先提取 fenced code block
        m = re.search(r"```python\s*(.*?)```", text, flags=re.S | re.I)
        if m:
            return m.group(1).strip()

        # 其次提取 [PYTHON_CODE] section
        code_text = self._extract_section(text, "PYTHON_CODE")
        if code_text:
            code_text = re.sub(r"^```(?:python)?\s*", "", code_text, flags=re.I)
            code_text = re.sub(r"\s*```$", "", code_text)
            return code_text.strip()

        return ""