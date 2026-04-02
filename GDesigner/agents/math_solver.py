from typing import List,Any,Dict

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
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        route_context = kwargs.get("route_context", {}) or {}
        memory_summary = kwargs.get("memory_summary", {}) or {}

        selected_packets = select_context_packets(
            role=self.role,
            spatial_info=spatial_info,
            temporal_info=temporal_info,
        )
        packet_context = render_role_specific_packet_context(self.role, selected_packets)
        if packet_context:
            packet_context = f"[Structured predecessor context]\n{packet_context}\n"

        memory_packets = build_memory_packets(memory_summary, top_k=2)
        memory_context = format_memory_context(memory_packets)

        route_str = (
            f"[Route context]\n"
            f"- step_idx: {route_context.get('step_idx', -1)}\n"
            f"- visible_predecessors: {route_context.get('visible_predecessor_ids', [])}\n"
        )

        if packet_context:
            user_prompt += "\n\n" + route_str + "\n" + packet_context
        else:
            user_prompt += "\n\n" + route_str + "\n[Visible collaboration packets]\n- None"

        if memory_context:
            user_prompt += "\n\n" + memory_context

        user_prompt += "\n\n" + self._get_role_task_block()
        user_prompt += "\n\n" + self._get_role_output_format_block()

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
                "Check whether the current derivation and candidate answer are valid.\n"
                "Your main job is to identify errors, missing steps, unsupported jumps, or invalid formulas.\n"
                "You do not need to fully solve the problem unless necessary.\n"
                "If the solution is flawed, provide actionable revision feedback for MathSolver.\n"
                "Output a clear verdict: pass, revise, or fail."
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

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        if self.role == "ProgrammingExpert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"
        structured = self._extract_structured_packet(response, self.role)
        self.output_packet = build_output_packet(self.role, response, structured=structured)
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

        self.output_packet = build_output_packet(self.role, response, structured=structured)
        print(f"#################system_prompt:{system_prompt}")
        print(f"#################user_prompt:{user_prompt}")
        print(f"#################response:{response}")
        print(f"#################packet:{self.output_packet}")
        return response
    
    def _extract_section(self, text: str, section_name: str) -> str:
        pattern = rf"^\s*\[{re.escape(section_name)}\]\s*(.*?)(?=^\s*\[[A-Z_]+\]|\Z)"
        m = re.search(pattern, text, flags=re.S | re.M)
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
                "known_facts": used_packets,
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
            verdict = self._extract_section(text, "VERDICT")
            errors = self._extract_bullets(self._extract_section(text, "ERRORS"))
            correct_answer = self._extract_section(text, "CORRECT_ANSWER")
            final_text = self._extract_section(text, "FINAL")

            if not correct_answer and verdict.strip().lower() == "correct":
                correct_answer = target_answer

            packet.update({
                "summary": summary or self._truncate(text, 240),
                "known_facts": [target_answer] if target_answer else [],
                "plan_steps": recomputation,           # 兼容旧下游
                "candidate_answer": correct_answer.strip() if correct_answer else "",
                "verdict": verdict.strip() if verdict else "",
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