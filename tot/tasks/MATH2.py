import re
import os
import json
import csv
import random
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.MATH2 import *
from tot.aggregated_skills import aggregated_skills
from tot.models import get_output

class Math2Task(Task):
    def __init__(self):
        super().__init__()
        self.data = []
        self.value_cache = {}

        path = os.path.join(DATA_PATH, 'MATH2', 'math2_subset100.csv')
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                problem = row[2]  # Third column is the problem
                solution = row[3]  # Fourth column is the solution
                self.data.append({'problem': problem, 'solution': solution})

        self.aggregated_skills = aggregated_skills

        self.skill_examples = {}
        skill_examples_path = os.path.join(DATA_PATH, 'skill_examples')
        for skill in self.aggregated_skills.values():
            skill_file = os.path.join(skill_examples_path, f'math_train_{skill}_with_examples.jsonl')
            if os.path.exists(skill_file):
                with open(skill_file, 'r', encoding='utf-8') as f:
                    examples = [json.loads(line) for line in f]
                    self.skill_examples[skill] = examples

        self.steps = 8  # Set heuristically
        self.stops = ['\n'] * self.steps

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['problem']

    @staticmethod
    def extract_from_text(text: str, prefixes: list) -> str:
        # Searches the text for the first occurrence of any of the prefixes and returns the content after it
        for prefix in prefixes:
            pattern = re.escape(prefix) + r'\s*(.*)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ''

    def test_output(self, idx: int, output: str, model: str):
        problem = self.data[idx]['problem']
        correct_solution = self.data[idx]['solution']
        model_solution = self.extract_from_text(output, ['Answer:'])

        # Use LLM-as-a-judge to judge correctness
        is_correct = self.llm_judge(problem, correct_solution, model_solution, model)
        print("Correctness judged by LLM: ", is_correct)
        return {'r': int(is_correct)}

    def llm_judge(self, problem: str, correct_solution: str, model_solution: str, model: str) -> bool:
        prompt = judge_prompt.format(
            problem=problem,
            correct_solution=correct_solution,
            model_solution=model_solution
        )
        response = get_output(prompt, model=model)[0]
        judgement = self.extract_from_text(response, ['Judgement:'])
        if judgement:
            judgement = judgement.strip().lower()
            return 'correct' in judgement
        else:
            return False

    def propose_prompt_wrap(self, problem: str, model: str, previous_step: str = '') -> str:
        if previous_step.strip():
            skill_prompt = skill_identification_prompt.format(problem=problem, aggregated_skills=self.aggregated_skills, previous_step = previous_step)
            skill_response = get_output(skill_prompt, model=model)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            prompt = propose_with_skill_prompt.format(
                problem=problem,
                previous_step=previous_step,
                skill=skill,
                in_context_example=in_context_example
            )
        else:
            # No previous steps; use starting prompt
            skill_prompt = skill_identification_prompt_start.format(problem=problem, aggregated_skills=self.aggregated_skills)
            skill_response = get_output(skill_prompt, model=model)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            prompt = start_with_skill_prompt.format(
                problem=problem,
                skill=skill,
                in_context_example=in_context_example
            )
        return prompt

    def value_prompt_wrap(self, x: str, y: str) -> str:
        answer = self.extract_from_text(y, ['Answer:'])
        if answer:
            prompt = value_last_step_prompt.format(problem=x, answer=answer)
        else:
            step = self.extract_from_text(y, ['First step:', 'Possible next step:'])
            if not step:
                step = y.strip()
            prompt = value_prompt.format(problem=x, current_step=step)
        return prompt

    def value_outputs_unwrap(self, value_outputs: list) -> float:
        value_names = []
        for output in value_outputs:
            evaluation = self.extract_from_text(output, ['Evaluation:', 'Judgement:'])
            if evaluation:
                value_names.append(evaluation.strip().lower())
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value_map.get(name, 0) for name in value_names)
        return value

    def get_in_context_example(self, skill: str) -> str:
        examples = self.skill_examples.get(skill, [])
        if examples:
            example = random.choice(examples)
            example_problem = example['problem']
            example_solution = example['solution']
            in_context_example = f"Example Problem:\n{example_problem}\n\nExample Solution:\n{example_solution}\n"
        else:
            in_context_example = ''
        return in_context_example