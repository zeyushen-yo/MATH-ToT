standard_prompt = '''Solve the following problem.

Problem: 
{problem}

Provide your answer in the following format: "Answer: [Your answer]".
'''

cot_prompt = '''Solve the following problem by thinking step by step.

Problem: 
{problem}

Provide your answer in the following format: "Answer: [Your answer]".
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

start_with_skill_prompt = '''Using the identified skill "{skill}", propose a possible first step to solve the following problem. You can refer to the example problem and its solution for guidance.

Problem: 
{problem}

Example problem applying the identified skill:
{in_context_example}

Provide your proposal in the following format: "First step: [Your first step]".
'''

propose_with_skill_prompt = '''Using the identified skill "{skill}", provide the most likely next step to solve the problem given your current step. You can refer to the example problem and its solution for guidance. 

If the current step leads to an answer / already contains an answer, provide the answer in the required format.

Problem:
{problem}

Your current step:
{previous_step}

Example problem applying the identified skill:
{in_context_example}

If you have found the answer, provide your answer in the following format: "Answer: [Your answer]".

Otherwise, provide your next step in the following format: "Possible next step: [Your possible next step]".
'''

propose_without_skill_prompt = '''Provide the most likely next step to solve the problem given your current step.

If the current step leads to an answer / already contains an answer, provide the answer in the required format.

Problem:
{problem}

Your current step:
{previous_step}

If you have found the answer, provide your answer in the following format: "Answer: [Your answer]".

Otherwise, provide your next step in the following format: "Possible next step: [Your possible next step]".
'''

value_prompt = '''Evaluate the usefulness and correctness of the following step in solving the problem.

Problem:
{problem}

Current Step:
{current_step}

First, provide reasoning about the step's validity and potential to lead to a solution. Then, provide an overall evaluation in one word from the following options: impossible, likely, sure.

Provide your evaluation in the following format: "Evaluation: [impossible/likely/sure]".
'''

value_last_step_prompt = '''Evaluate the likelihood of the following answer in being the correct answer to the problem.

Input: 
{problem}

Answer: 
{answer}

First, explicitly go through some sanity checks on the answer. Then, provide an overall evaluation in one word from the following options: impossible, likely, sure.

Provide your evaluation in the following format: "Evaluation: [impossible/likely/sure]".
'''

judge_prompt = '''Given the following problem and two solutions, determine whether the second solution is correct with respect to the problem.

Problem:
{problem}

Correct Solution:
{correct_solution}

Model's Solution:
{model_solution}

As long as the model's solution contains an answer that is mathematically equivalent to the correct solution, it should be judged as correct. There might be irrevalent characters around the model's solution, and you should disregard them.
However, the model's solution does have to explicit contain such an answer in order to be judged as correct. It cannot be an intermediate step without a real solution.

Provide your judgement in the following format: "Judgement: [correct/wrong]".
'''