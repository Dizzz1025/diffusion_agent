import yaml
from typing import List,Any,Dict

from ..graph.node import Node
from .agent_registry import AgentRegistry
from ..llm.llm_registry import LLMRegistry
from ..prompt.prompt_set_registry import PromptSetRegistry
from ..tools.coding.python_executor import PyExecutor
from ..utils.const import GDesigner_ROOT

def _render_packet_for_reference(info: Dict[str, Any]) -> str:
    role = info.get("role", "Unknown")
    packet = info.get("packet") or {}
    packet_type = packet.get("packet_type", "generic")

    if not packet:
        raw = (info.get("output") or "").strip()
        return f"role={role}, raw_output={raw}"

    lines = [f"role={role}, packet_type={packet_type}"]

    if packet.get("summary"):
        lines.append(f"summary: {packet['summary']}")

    if packet.get("known_facts"):
        lines.append("known_facts:")
        for fact in packet["known_facts"][:4]:
            lines.append(f"- {fact}")

    if packet.get("plan_steps"):
        lines.append("plan_steps:")
        for step in packet["plan_steps"][:4]:
            lines.append(f"- {step}")

    if packet.get("candidate_answer"):
        lines.append(f"candidate_answer: {packet['candidate_answer']}")

    if packet.get("verdict"):
        lines.append(f"verdict: {packet['verdict']}")

    if packet.get("errors_found"):
        lines.append("errors_found:")
        for err in packet["errors_found"][:4]:
            lines.append(f"- {err}")

    if packet.get("code_result"):
        lines.append(f"code_result: {packet['code_result']}")

    if packet.get("final_text"):
        lines.append(f"final_text: {packet['final_text']}")

    return "\n".join(lines)


def _build_final_reference_context(spatial_info: Dict[str, Any]) -> str:
    blocks = []
    for node_id, info in spatial_info.items():
        block = _render_packet_for_reference(info)
        blocks.append(f"{node_id}:\n{block}")
    return "\n\n".join(blocks).strip()

def _packet_preferred_answer(info: Dict[str, Any]) -> str:
    packet = info.get("packet") or {}
    packet_type = packet.get("packet_type", "")
    verdict = str(packet.get("verdict", "") or "").strip().lower()

    # checker 的结论优先级最高
    if packet_type == "check":
        if verdict == "correct" and packet.get("candidate_answer"):
            return str(packet["candidate_answer"]).strip()
        if packet.get("correct_answer"):
            return str(packet["correct_answer"]).strip()
        if packet.get("candidate_answer"):
            return str(packet["candidate_answer"]).strip()

    # programmer 优先取执行结果，其次取候选答案
    if packet_type == "program":
        if packet.get("code_result"):
            return str(packet["code_result"]).strip()
        if packet.get("candidate_answer"):
            return str(packet["candidate_answer"]).strip()

    # solver 直接取候选答案
    if packet_type == "solve":
        if packet.get("candidate_answer"):
            return str(packet["candidate_answer"]).strip()

    return ""


def _fallback_processed_answer(prompt_set, info: Dict[str, Any]) -> str:
    raw_output = info.get("output", "") or ""
    try:
        return str(prompt_set.postprocess_answer(raw_output)).strip()
    except Exception:
        return raw_output.strip()

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        spatial_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):  # is python code
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}.\n At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name, model_name=llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{self.role}.\n {self.constraint}"

        spatial_str = _build_final_reference_context(spatial_info)

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} "
            f"The task is:\n\n{raw_inputs['task']}.\n"
            f"At the same time, the structured outputs of other agents are as follows:\n\n"
            f"{spatial_str}"
        )
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        print(f"################system prompt:{system_prompt}")
        print(f"################user prompt:{user_prompt}")
        print(f"################response:{response}")
        return response

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
                
    def _execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        candidates = []

        for info in spatial_info.values():
            ans = _packet_preferred_answer(info)
            if not ans:
                ans = _fallback_processed_answer(self.prompt_set, info)
            if ans:
                candidates.append(ans)

        return candidates[-1] if candidates else ""
    
    async def _async_execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        candidates = []

        for info in spatial_info.values():
            ans = _packet_preferred_answer(info)
            if not ans:
                ans = _fallback_processed_answer(self.prompt_set, info)
            if ans:
                candidates.append(ans)

        return candidates[-1] if candidates else ""


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
    
    def _execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        output_num = {}
        max_output = ""
        max_output_num = 0

        for info in spatial_info.values():
            processed_output = _packet_preferred_answer(info)
            if not processed_output:
                processed_output = _fallback_processed_answer(self.prompt_set, info)

            processed_output = str(processed_output).strip()
            if not processed_output:
                continue

            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1

            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]

        return max_output
    
    async def _async_execute(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        output_num = {}
        max_output = ""
        max_output_num = 0

        for info in spatial_info.values():
            processed_output = _packet_preferred_answer(info)
            if not processed_output:
                processed_output = _fallback_processed_answer(self.prompt_set, info)

            processed_output = str(processed_output).strip()
            if not processed_output:
                continue

            print(processed_output)

            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1

            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]

        return max_output
