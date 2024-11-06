naive_prompt = '''Solve the following problem.
Input Problem: {input_problem}
Provide your answer in terms of a single expression in the following format: "Answer: [Your answer]".
'''

starting_prompt = '''Propose a possible first-step that can help you solve the following problem. Providing a single step is sufficient; you don't need to solve the problem immediately, unless you are sure that the current step can lead to the correct answer.
Input Problem: {input_problem}
Provide your proposal in the following format: "First step: [Your first step]".
If you have found the answer, provide your answer in terms of a single expression in the following format: "Answer: [Your answer]".
'''

value_prompt = '''Given an input problem and a current step, evaluate whether you can solve the problem with the current step (provide a one-word evaluation from the following three options: sure/likely/impossible).
Input Problem: {input_problem}
Your current step: {input_step}
First explicitly go through some reasonings for your evaluation. You can consider two aspects: whether the current step is correct, and whether this step can lead you to the ultimate answer. 
To check whether the current step is correct, you can check whether there are any computational or logical issues. To check whether this step can lead you to the ultimate answer, you can study some simple versions or special cases of the problem and see whether the current step could work.
You don't have to do both checks; check whatever you find necessary.
Then, provide your evaluation in the following format "Evaluation: [Your evaluation]".
'''

propose_prompt = '''Provide the most likely next step to solve the problem given your current step. If the current step leads to an answer, provide the answer.
Input Problem: {input_problem}
Your current step: {input_step}
If you haven't found the answer, provide your next-step proposal in the following format "Possible next step: [Your possible next step]".
If you have found the answer, provide your answer in terms of a single expression in the following format: "Answer: [Your answer]".
'''

value_last_step_prompt = '''Given an input problem and an answer, do some sanity checks on the answer. Then, provide a judgement for whether the answer is correct (provide a one-word judgement from the following two options: sure/impossible).
Input: {input_problem}
Answer: {answer}
First explicitly go through some sanity checks. Then, provide your judgement in the following format: "Judgement: [Your judgement]".
'''