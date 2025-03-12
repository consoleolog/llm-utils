import re
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import List, Dict, Optional
from langchain_core.language_models import BaseChatModel

def extract_xml(text: str, tag:str)->str:
    match = re.search(f"<{tag}\\s*>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""

def _parse_tasks(tasks_xml: str)-> List[Dict]:
    tasks = []
    current_task = {}
    for line in tasks_xml.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[12:-13].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)
    return tasks

def _format_prompt(template:str, **kwargs)->str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing Required Prompt Variable: {e}")

class LLMUtils:
    def __init__(self, model: BaseChatModel = None):
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model

    def call(self, prompt:str, system_prompt: str = "")->str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ] if system_prompt != "" else [{"role": "user", "content": prompt}]
        return self.model.invoke(messages).content

    def chain(self, inputs:str, prompts: List[str])->str:
        result = inputs
        for i, prompt in enumerate(prompts, 1):
            result = self.call(f"{prompt}\nInput: {result}")
            pprint(f"Step {i}\n{result}")
        return result

    def parallel(self, prompt:str, inputs: List[str], n_workers:int=3)->List[str]:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.call, f"{prompt}\nInput: {x}", "") for x in inputs]
        return [f.result() for f in futures]

    def route(self, inputs:str, routes: Dict[str, str])->str:
        pprint(f"\nAvailable routes: {list(routes.keys())}")
        selector_prompt = f"""
        Input 을 분석 하고 routes 중에서 가장 적절한 route 를 선택 해줘
        routes : {list(routes.keys())}

        먼저 route 를 선택 하는 이유를 설명 해주고, 다음과 같은 XML Format 으로 응답 해줘
        XML Format :
        <reasoning>
        Input 이 왜 이 route 로 routing 이 되어야 하는지 이유 설명
        Consider key terms, user intent, and urgency level.
        </reasoning>        

        <selection>
        선택한 route 이름
        </selection>

        Input: {inputs}
        """.strip()
        route_response = self.call(prompt=selector_prompt)
        reasoning = extract_xml(route_response, "reasoning")
        route_key = extract_xml(route_response, "selection").strip().lower()
        pprint(f"\nRouting 분석: \n{reasoning}\n Selected Route: {route_key}\n")
        selector_prompt = routes[route_key]
        return self.call(prompt=f"{selector_prompt}\nInput: {inputs}")

    def generate(self, prompt:str, task:str, context:str = "")->tuple[str, str]:
        full_prompt = f"{prompt}\n{context}\nTask: {task}" if context else f"{prompt}\nTask: {task}"
        response = self.call(full_prompt)
        thoughts = extract_xml(response, "thoughts")
        result = extract_xml(response, "response")
        pprint(f"\n=== GENERATION START ===\nThoughts:\n{thoughts}\nGenerated:\n{result}\n=== GENERATION END ===\n")
        return thoughts, result

    def evaluate(self, prompt: str, context: str, task: str) -> tuple[str, str]:
        full_prompt = f"{prompt}\nOriginal task: {task}\nContext to evaluate: {context}"
        response = self.call(full_prompt)
        evaluation = extract_xml(response, "evaluation")
        feedback = extract_xml(response, "feedback")
        pprint(f"=== EVALUATION START ===\nStatus: {evaluation}\nFeedback: {feedback}\n=== EVALUATION END ===\n")
        return evaluation, feedback


    def loop(self, task: str, evaluator_prompt: str, generator_prompt: str) -> tuple[str, list[dict]]:
        memory = []
        chain_of_thought = []

        thoughts, result = self.generate(generator_prompt, task)
        memory.append(result)
        chain_of_thought.append({
            "thoughts": thoughts,
            "result": result
        })

        while True:
            evaluation, feedback = self.evaluate(evaluator_prompt, result, task)
            if evaluation == "PASS":
                return result, chain_of_thought

            context = "\n".join([
                "Previous attempts:",
                *[f"- {m}" for m in memory],
                f"\nFeedback: {feedback}"
            ])

            thoughts, result = self.generate(generator_prompt, task, context)
            memory.append(result)
            chain_of_thought.append({
                "thoughts": thoughts,
                "result": result
            })

    def orchestrator(self, orchestrator_prompt: str, worker_prompt: str):
        return self._FlexibleOrchestrator(orchestrator_prompt, worker_prompt)

    def process(self, task: str, context: Optional[Dict] = None, orchestrator = None):
        context = context or {}
        orchestrator_input = _format_prompt(
            orchestrator.orchestrator_prompt,
            task=task,
            **context
        )
        orchestrator_response = self.call(orchestrator_input)
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = _parse_tasks(tasks_xml)
        pprint(f"\n=== ORCHESTRATOR OUTPUT === \nANALYSIS:\n{analysis} \nTASKS:\n {tasks}")
        worker_results = []
        for task_info in tasks:
            worker_input = _format_prompt(
                orchestrator.worker_prompt,
                original_task=task,
                task_type=task_info["type"],
                task_description=task_info["description"],
                **context
            )
            worker_response = self.call(worker_input)
            result = extract_xml(worker_response, "response")
            worker_results.append({
                "type": task_info["type"],
                "description": task_info["description"],
                "result": result
            })
            pprint(f"\n=== WORKER RESULT ({task_info['type']}) ===\n{result}\n")
        return {
            "analysis": analysis,
            "worker_results": worker_results
        }

    class _FlexibleOrchestrator:
        def __init__(self, orchestrator_prompt: str, worker_prompt: str,):
            self.orchestrator_prompt = orchestrator_prompt
            self.worker_prompt = worker_prompt

