import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.MATH import *
from tot.data.MATH.math_equivalence import is_equiv
from tot.aggregated_skills import aggregated_skills
from tot.models import get_output

class MathTask(Task):
    def __init__(self):
        super().__init__()
        self.data = []
        self.value_cache = {}

        path = os.path.join(DATA_PATH, 'MATH', 'test')
        self.aggregated_skills = aggregated_skills

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        problem_data = json.load(f)
                        self.data.append(problem_data)

        self.skill_examples = {}
        skill_examples_path = os.path.join(DATA_PATH, 'skill_examples')
        for skill in self.aggregated_skills.values():
            skill_file = os.path.join(skill_examples_path, f'math_train_{skill}_with_examples.jsonl')
            if os.path.exists(skill_file):
                with open(skill_file, 'r', encoding='utf-8') as f:
                    examples = [json.loads(line) for line in f]
                    self.skill_examples[skill] = examples

        self.steps = 5  # set heuristically
        self.stops = ['\n'] * self.steps

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['problem']

    def test_output(self, idx: int, output: str, model: str):
        solution = self.data[idx]['solution']
        correct_answer = self.extract_answer(solution)
        model_answer = self.extract_answer(output)
        print("correct answer: ", correct_answer)
        print("model answer: ", model_answer)

        if correct_answer is None or model_answer is None:
            return {'r': 0}
        else:
            return {'r': is_equiv(correct_answer.strip(), model_answer.strip())}

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

    # GPT might add punctuations after "Answer: "; make sure to remove such things
    @staticmethod
    def extract_number_or_expression(text):
        text = text.strip().strip('.,;:')
        pattern = r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$|^[A-Za-z][A-Za-z0-9_+\-*/^()\s=]*$'
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if re.match(pattern, line):
                return line
        return None

    @staticmethod
    def extract_answer(text):
        # get GPT answer
        answer_text = MathTask.extract_from_text(text, ['Answer:'])
        if answer_text:
            answer = MathTask.extract_number_or_expression(answer_text)
            if answer:
                return answer.strip()
        
        # get correct answer
        match = re.search(r'\\boxed\{(.+?)\}', text)
        if match:
            return match.group(1).strip()
        else:
            # If no \box, extract the last number in the text
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
            if numbers:
                return numbers[-1].strip()
            else:
                return None

    @staticmethod
    def starting_prompt_wrap(x: str, y:str='') -> str:
        return starting_prompt.format(problem=x) + y

    @staticmethod
    def naive_prompt_wrap(x: str, y:str='') -> str:
        return naive_prompt.format(problem=x) + y
    
    def propose_prompt_wrap(self, apply_skills: bool, decompose_problem: bool, problem: str, model: str, previous_step: str = '') -> str:
        if previous_step.strip():
            if apply_skills:
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
                prompt = propose_without_skill_prompt.format(
                    problem=problem,
                    previous_step=previous_step
                )
        else:
            # No previous steps; use starting prompt
            if apply_skills:
                skill_prompt = skill_identification_prompt_start.format(problem=problem, aggregated_skills=self.aggregated_skills)
                skill_response = get_output(skill_prompt, model=model)[0]
                skill = self.extract_from_text(skill_response, ['Skill:']).strip()

                in_context_example = self.get_in_context_example(skill)
                prompt = start_with_skill_prompt.format(
                    problem=problem,
                    skill=skill,
                    in_context_example=in_context_example
                )
            else:
                prompt = start_without_skill_prompt.format(
                    problem=problem,
                    previous_step=previous_step
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
    
    def standard_prompt_wrap(self, x: str, y:str, apply_skills:bool, model:str) -> str:
        if apply_skills:
            skill_prompt = skill_identification_prompt_start.format(problem=x, aggregated_skills=self.aggregated_skills)
            skill_response = get_output(skill_prompt, model=model)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            return standard_with_skill_prompt.format(problem=x, skill=skill, in_context_example=in_context_example) + y
        else:
            return standard_prompt.format(problem=x) + y

    def cot_prompt_wrap(self, x: str, y:str, apply_skills:bool, model:str) -> str:
        if apply_skills:
            skill_prompt = skill_identification_prompt_start.format(problem=x, aggregated_skills=self.aggregated_skills)
            skill_response = get_output(skill_prompt, model=model)[0]
            skill = self.extract_from_text(skill_response, ['Skill:']).strip()

            in_context_example = self.get_in_context_example(skill)
            return cot_with_skill_prompt.format(problem=x, skill=skill, in_context_example=in_context_example) + y
        else:
            return cot_prompt.format(problem=x) + y