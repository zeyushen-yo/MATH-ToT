standard_prompt = '''Solve the following problem.

Problem: 
{problem}

Provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".
'''

cot_prompt = '''Solve the following problem by thinking step by step.

Problem: 
{problem}

Provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".
'''

standard_with_skill_prompt = '''Using the identified skill "{skill}", solve the following problem. You can refer to the example problem and its solution for guidance. 

Problem: 
{problem}

Example problem applying the identified skill:
{in_context_example}

Provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".
'''

cot_with_skill_prompt = '''Using the identified skill "{skill}", solve the following problem by thinking step by step. You can refer to the example problem and its solution for guidance.

Problem: 
{problem}

Example problem applying the identified skill:
{in_context_example}

Provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".
'''

skill_identification_prompt_start = '''Here is a list of skills:\n {aggregated_skills} \n for solving mathematical problems.

Identify the most relevant mathematical skill from the list that can be used to solve the following problem. You must name the skill in exactly the same way as it appears in the list

Problem: 
{problem}

Provide your answer in the following format: "Skill: [Your identified skill]".
'''

skill_identification_prompt = '''Here is a list of skills:\n {aggregated_skills} \n for solving mathematical problems.

Given your current step, identify the most relevant mathematical skill from the list that can be used to solve the following problem. You must name the skill in exactly the same way as it appears in the list

Problem: 
{problem}

Your current step:
{previous_step}

Provide your answer in the following format: "Skill: [Your identified skill]".
'''

start_with_skill_prompt = '''Using the identified skill "{skill}", propose a possible first step to solve the following problem. You can refer to the example problem and its solution for guidance. Your step needs to be concrete. In other words, you not only need to propose what you can do, but you also need to show how you can do it.

Problem: 
{problem}

Example problem applying the identified skill:
{in_context_example}

Provide your proposal in the following format: "First step: [Your first step]".
'''

start_without_skill_prompt = '''Propose a possible first step to solve the following problem. Your step needs to be concrete. In other words, you not only need to propose what you can do, but you also need to show how you can do it.

Problem: 
{problem}

Provide your proposal in the following format: "First step: [Your first step]".
'''

propose_with_skill_prompt = '''Using the identified skill "{skill}", provide the most likely next step to solve the problem given your current step. You can refer to the example problem and its solution for guidance. Your step needs to be concrete. In other words, you not only need to propose what you can do, but you also need to show how you can do it.

If the current step leads to an answer / already contains an answer, provide the answer in the required format.

Problem:
{problem}

Your current step:
{previous_step}

Example problem applying the identified skill:
{in_context_example}

If you have found the answer, provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".

Otherwise, provide your next step in the following format: "Possible next step: [Your possible next step]".
'''

propose_without_skill_prompt = '''Provide the most likely next step to solve the problem given your current step. Your step needs to be concrete. In other words, you not only need to propose what you can do, but you also need to show how you can do it.

If the current step leads to an answer / already contains an answer, provide the answer in the required format.

Problem:
{problem}

Your current step:
{previous_step}

If you have found the answer, provide your answer in terms of a single number or expression in the following format: "Answer: [Your answer]".

Otherwise, provide your next step in the following format: "Possible next step: [Your possible next step]".
'''

value_prompt = '''Evaluate the usefulness and correctness of the following step in solving the problem.

Problem:
{problem}

Current Step:
{current_step}

First, explicitly check whether the computation and logic in this step are correct. Then, provide an overall evaluation in one word from the following options: impossible, likely, sure.

Provide your evaluation in the following format: "Evaluation: [impossible/likely/sure]".
'''

value_last_step_prompt = '''Evaluate the likelihood of the following answer in being the correct answer to the problem.

Input: 
{problem}

Answer: 
{answer}

First, explicitly go through some sanity checks on the answer. Then, provide an overall evaluation in one word from the following options: impossible, likely, sure.

Provide your evaluation in the following format: "Evaluation: [impossible/likely/sure]".