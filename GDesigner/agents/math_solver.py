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
    format_memory_context,
)
import re

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: str | None =None, role:str = None ,domain: str = "", llm_name: str = "",):
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
        packet_context = format_packet_context(selected_packets)

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

        # 给不同角色再加一层明确要求
        if self.role == "ProblemDecomposer":
            user_prompt += (
                "\n\n[Your required output]\n"
                "Return a decomposition only: variables, plan_steps, equations.\n"
                "Do not solve the problem completely."
            )
        elif self.role == "MathSolver":
            user_prompt += (
                "\n\n[Your required output]\n"
                "Use the available plan/program/check packets if useful. "
                "Provide derivation and the final numerical answer in format '####Answer: <number>'."
            )
        elif self.role == "ProgrammingExpert":
            user_prompt += (
                "\n\n[Your required output]\n"
                "Write Python code only if useful, then report the program result."
            )
        elif self.role == "CalculationChecker":
            user_prompt += (
                "\n\n[Your required output]\n"
                "Verify the candidate answer and key computations. "
                "State whether the answer is correct, incorrect, or uncertain."
            )

        output_format_block = """
        [Output format]
        Please structure your response using the following sections when applicable.

        [SUMMARY]
        A short summary of your reasoning.

        [KNOWN_FACTS]
        List the known quantities, variables, or facts you used.
        Use bullet points if possible.

        [PLAN]
        List the main steps if you are decomposing or solving.
        Use bullet points if possible.

        [CANDIDATE_ANSWER]
        Give the candidate numerical answer if available.

        [VERDICT]
        State whether a previous answer is correct or incorrect if you are checking.

        [ERRORS]
        List any specific errors you found.
        Use bullet points if possible.

        [CODE_RESULT]
        If you used code, report the computed result.

        [FINAL]
        Your final conclusion for this role.
        """

        user_prompt += "\n\n" + output_format_block

        return system_prompt, user_prompt
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        if self.role == "ProgrammingExpert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"

        self.output_packet = build_output_packet(self.role, response)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info, **kwargs)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        if self.role == "ProgrammingExpert":
            answer = execute_code_get_return(response.lstrip("```python\n").rstrip("\n```"))
            response += f"\nthe answer is {answer}"

        self.output_packet = build_output_packet(self.role, response)
        print(f"#################system_prompt:{system_prompt}")
        print(f"#################user_prompt:{user_prompt}")
        print(f"#################response:{response}")
        print(f"#################packet:{self.output_packet}")
        return response
    
    def _extract_section(self, text: str, section_name: str) -> str:
        patterns = [
            rf"\[{re.escape(section_name)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)",
            rf"\*\*{re.escape(section_name)}\*\*\s*(.*?)(?=\n\*\*[A-Z_]+\*\*|\Z)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, flags=re.S)
            if m:
                return m.group(1).strip()
        return ""
    
    def _extract_bullets(self, text: str) -> List[str]:
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("-"):
                lines.append(line[1:].strip())
            elif re.match(r"^\d+\.", line):
                lines.append(line)
        return lines
    
    def _extract_structured_packet(self, text: str, role: str) -> Dict[str, Any]:
        summary = self._extract_section(text, "SUMMARY")
        known_facts_text = self._extract_section(text, "KNOWN_FACTS")
        plan_text = self._extract_section(text, "PLAN")
        candidate_answer = self._extract_section(text, "CANDIDATE_ANSWER")
        verdict = self._extract_section(text, "VERDICT")
        errors_text = self._extract_section(text, "ERRORS")
        code_result = self._extract_section(text, "CODE_RESULT")
        final_text = self._extract_section(text, "FINAL")

        known_facts = self._extract_bullets(known_facts_text)
        plan_steps = self._extract_bullets(plan_text)
        errors_found = self._extract_bullets(errors_text)

        if not candidate_answer:
            candidate_answer = self._extract_answer_hint(text)

        packet = {
            "role": role,
            "summary": summary or self._truncate(text, 240),
            "known_facts": known_facts,
            "plan_steps": plan_steps,
            "candidate_answer": candidate_answer.strip() if candidate_answer else "",
            "verdict": verdict.strip() if verdict else "",
            "errors_found": errors_found,
            "code_result": code_result.strip() if code_result else "",
            "final_text": final_text.strip() if final_text else "",
        }
        return packet
    
    