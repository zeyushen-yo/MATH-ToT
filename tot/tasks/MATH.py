import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.MATH import *
from tot.data.MATH.math_equivalence import is_equiv

class MathTask(Task):
    def __init__(self):
        super().__init__()
        self.data = []
        self.value_cache = {}

        # path = os.path.join(DATA_PATH, 'MATH', 'test')
        # a small subset of test dataset
        path = os.path.join(DATA_PATH, 'MATH', 'example')

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        problem_data = json.load(f)
                        self.data.append(problem_data)

        self.steps = 5  # set heuristically
        self.stops = ['\n'] * self.steps

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]['problem']

    def test_output(self, idx: int, output: str):
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
        return starting_prompt.format(input_problem=x) + y

    @staticmethod
    def naive_prompt_wrap(x: str, y:str='') -> str:
        return naive_prompt.format(input_problem=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str = '') -> str:
        if y.strip():
            step = MathTask.extract_from_text(y, ['First step:', 'Possible next step:'])
            if not step:
                step = y.strip()
            prompt = propose_prompt.format(input_problem=x, input_step=step)
        else:
            # No previous steps; use the starting prompt
            prompt = starting_prompt.format(input_problem=x)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        answer = MathTask.extract_from_text(y, ['Answer:'])
        if answer:
            prompt = value_last_step_prompt.format(input_problem=x, answer=answer)
        else:
            step = MathTask.extract_from_text(y, ['First step:', 'Possible next step:'])
            if not step:
                step = y.strip()
            prompt = value_prompt.format(input_problem=x, input_step=step)
        return prompt
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_names = []
        for output in value_outputs:
            evaluation = MathTask.extract_from_text(output, ['Evaluation:', 'Judgement:'])
            if evaluation:
                value_names.append(evaluation.strip().lower())
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value_map.get(name, 0) for name in value_names)
        return value