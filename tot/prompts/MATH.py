starting_prompt = '''Propose a possible first-step that can help you solve the following problem. Providing a single step is sufficient; you don't need to solve the problem immediately, unless you are sure that the current step can lead to the correct answer.
Input: {input_problem}
Provide your proposal in the following format: "First step: [Your first step]".
If you find the answer, provide your answer in terms of a single number in the following format: "Answer: [Your answer]".
'''

value_prompt = '''Evaluate whether you can solve the problem with the current step (provide a one-word answer from the following three options: sure/likely/impossible).
Input Problem: {input_problem}
Your current step: {input_step}
Provide your answer in the following format "Evaluation: [Your evaluation]".
'''

propose_prompt = '''Provide the most likely next step to solve the problem given your current step. If the current step leads to an answer, provide the answer.
Input Problem: {input_problem}
Your current step: {input_step}
If you haven't found the answer, provide your next-step proposal in the following format "Possible next step: [Your possible next step]".
If you have found the answer, provide your answer in terms of a single number in the following format: "Answer: [Your answer]".
'''

value_last_step_prompt = '''Given an input problem and a solution, give a judgement if the answer is correct (provide a one-word answer from the following two options: sure/impossible).
Input: {input_problem}
Answer: {answer}
Provide your answer in the following format: "Judgement: [Your judgement]".
'''