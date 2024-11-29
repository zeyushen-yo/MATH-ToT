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

        self.steps = 10  # Set heuristically

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

    @staticmethod
    def extract_one_word_after_pattern(text: str, prefixes: list) -> str:
        for prefix in prefixes:
            pattern = re.escape(prefix) + r'\s*\W*(\w+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ''

    def extract_correct_answer(self, text):
        match = re.search(r'\\boxed\{(.+?)\}', text)
        if not match:
            return text
        else:
            return match.group(1).strip()
        
    def test_output(self, idx: int, output: str, model: str):
        problem = self.data[idx]['problem']
        correct_solution = self.extract_correct_answer(self.data[idx]['solution'])
        model_solution = self.extract_from_text(output, ['Answer:'])
        if not model_solution:
            model_solution = output

        print(model_solution)
        print(correct_solution)
        # Use LLM-as-a-judge to judge correctness
        is_correct = self.llm_judge(problem, correct_solution, model_solution, "o1-mini")
        print("Correctness judged by LLM: ", is_correct)
        return {'r': int(is_correct)}

    def llm_judge(self, problem: str, correct_solution: str, model_solution: str, model: str) -> bool:
        prompt = judge_prompt.format(
            problem=problem,
            correct_solution=correct_solution,
            model_solution=model_solution
        )
        response = get_output(prompt, model="o1-mini", temperature=1e-9)[0]
        judgement = self.extract_from_text(response, ['Judgement:'])
        if judgement:
            judgement = judgement.strip().lower()
            return 'correct' in judgement
        else:
            return False

    def simplify_problem(self, model, tokenizer, name, problem: str, previous_step: str, temperature: float) -> str:
        if previous_step.strip():
            prompt = simplify_problem_with_step_prompt.format(
                problem=problem,
                previous_step=previous_step
            )
        else:
            prompt = simplify_problem_prompt.format(
                problem=problem
            )
        simplified_problem = get_output(model, tokenizer, name, prompt, temperature=temperature)[0]
        simplified_problem = self.extract_from_text(simplified_problem, ['Simplified Problem:']).strip()
        return simplified_problem

    def solve_simplified_problem(self, model, tokenizer, name, simplified_problem: str, temperature: float) -> str:
        prompt = solve_simplified_problem_prompt.format(
            simplified_problem=simplified_problem
        )
        simplified_solution = get_output(model, tokenizer, name, prompt, temperature=temperature)[0]
        return simplified_solution

    def propose_prompt_wrap(self, model, tokenizer, name, apply_skills: bool, decompose_problem: bool, problem: str, temperature: float, previous_step: str = '') -> str:
        if previous_step.strip():
            if apply_skills:
                skill_prompt = skill_identification_prompt.format(problem=problem, aggregated_skills=self.aggregated_skills, previous_step=previous_step)
                skill_response = get_output(model, tokenizer, name, skill_prompt, temperature=temperature)[0]
                skill = self.extract_from_text(skill_response, ['Skill:']).strip()

                in_context_example = self.get_in_context_example(skill)
                prompt = propose_with_skill_prompt.format(
                    problem=problem,
                    previous_step=previous_step,
                    skill=skill,
                    in_context_example=in_context_example
                )
            elif decompose_problem:
                simplified_problem = self.simplify_problem(model, tokenizer, name, problem, previous_step, temperature)
                simplified_solution = self.solve_simplified_problem(model, tokenizer, name, simplified_problem, temperature)
                in_context_example = f"Simplified Problem:\n{simplified_problem}\n\n Solution to Simplified Problem:\n{simplified_solution}\n"
                prompt = propose_with_simplified_prompt.format(
                    problem=problem,
                    previous_step=previous_step,
                    in_context_example=in_context_example
                )
            else:
                prompt = propose_without_skill_prompt.format(
                    problem=problem,
                    previous_step=previous_step
                )
        else:
            # No previous steps; use starting prompt
            if apply_skills:
                skill_prompt = skill_identification_prompt_start.format(problem=problem, aggregated_skills=self.aggregated_skills)
                skill_response = get_output(model, tokenizer, name, skill_prompt, temperature=temperature)[0]
                skill = self.extract_from_text(skill_response, ['Skill:']).strip()

                in_context_example = self.get_in_context_example(skill)
                prompt = start_with_skill_prompt.format(
                    problem=problem,
                    skill=skill,
                    in_context_example=in_context_example
                )
            elif decompose_problem:
                simplified_problem = self.simplify_problem(model, tokenizer, name, problem, previous_step, temperature)
                simplified_solution = self.solve_simplified_problem(model, tokenizer, name, simplified_problem, temperature)
                in_context_example = f"Simplified Problem:\n{simplified_problem}\n\n Solution to Simplified Problem:\n{simplified_solution}\n"
                prompt = start_with_simplified_prompt.format(
                    problem=problem,
                    in_context_example=in_context_example
                )
            else:
                prompt = start_without_skill_prompt.format(
                    problem=problem
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
            evaluation = self.extract_one_word_after_pattern(output, ['Evaluation:', 'Judgement:'])
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

    def standard_prompt_wrap(self, model, tokenizer, name, x: str, y:str, apply_skills:bool, decompose_problem:bool, temperature:float) -> str:
        if apply_skills:
            skill_prompt = skill_identification_prompt_start.format(problem=x, aggregated_skills=self.aggregated_skills)
            skill_response = get_output(model, tokenizer, name, skill_prompt, temperature=temperature)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            return standard_with_skill_prompt.format(problem=x, skill=skill, in_context_example=in_context_example) + y
        elif decompose_problem:
            simplified_problem = self.simplify_problem(model, tokenizer, name, x, '', temperature)
            simplified_solution = self.solve_simplified_problem(model, tokenizer, name, simplified_problem, temperature)
            in_context_example = f"Simplified Problem:\n{simplified_problem}\n\n Solution to Simplified Problem:\n{simplified_solution}\n"  
            return standard_with_simplified_prompt.format(problem=x, in_context_example=in_context_example) + y          
        else:
            return standard_prompt.format(problem=x) + y

    def cot_prompt_wrap(self, model, tokenizer, name, x: str, y:str, apply_skills:bool, decompose_problem:bool, temperature:float) -> str:
        if apply_skills:
            skill_prompt = skill_identification_prompt_start.format(problem=x, aggregated_skills=self.aggregated_skills)
            skill_response = get_output(model, tokenizer, name, skill_prompt, temperature=temperature)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            return cot_with_skill_prompt.format(problem=x, skill=skill, in_context_example=in_context_example) + y
        elif decompose_problem:
            simplified_problem = self.simplify_problem(model, tokenizer, name, x, '', temperature)
            simplified_solution = self.solve_simplified_problem(model, tokenizer, name, simplified_problem, temperature)
            in_context_example = f"Simplified Problem:\n{simplified_problem}\n\n Solution to Simplified Problem:\n{simplified_solution}\n"  
            return cot_with_simplified_prompt.format(problem=x, in_context_example=in_context_example) + y              
        else:
            return cot_prompt.format(problem=x) + y