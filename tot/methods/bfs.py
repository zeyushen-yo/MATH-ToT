import itertools
import numpy as np
from functools import partial
from tot.models import get_output

def get_value(task, x, y, n_evaluate_sample, model, temperature, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = get_output(value_prompt, n=n_evaluate_sample, model=model, temperature=temperature)
    value = task.value_outputs_unwrap(value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, model, temperature, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, model, temperature, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_proposals(task, x, y, apply_skills, decompose_problem, n_generate_sample, model, temperature): 
    propose_prompt = task.propose_prompt_wrap(apply_skills, decompose_problem, x, model, y)
    proposals = get_output(propose_prompt, n=n_generate_sample, model=model, temperature=temperature)
    print(proposals)
    return proposals

def get_samples(task, x, y, n_generate_sample, prompt_sample, apply_skills, decompose_problem, model):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y, apply_skills, decompose_problem, model)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, apply_skills, decompose_problem, model)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = get_output(prompt, n=n_generate_sample)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    values = []
    infos = []
    ids = []
    step = 0
    all_ys_with_answers = []
    while True:
        # generation
        new_ys = [get_proposals(task, x, y, args.apply_skills, args.decompose_problem, args.n_generate_sample, args.backend, temperature=args.temperature) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        values = get_values(task, x, new_ys, args.n_evaluate_sample, args.backend, 0.1)

        selected_ids = [idx for idx in ids if values[idx] >= 1]

        if args.method_select == 'sample':
            if len(selected_ids) > args.n_select_sample:
                select_ids = np.random.choice(
                    selected_ids, size=args.n_select_sample, replace=False
                ).tolist()
            else:
                select_ids = selected_ids
        elif args.method_select == 'greedy':
            select_ids = sorted(
                selected_ids, key=lambda x: values[x], reverse=True
            )[:args.n_select_sample]

        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})

        # if the model is already sure about an answer, just output it
        cur_values = [values[idx] for idx in select_ids]
        ys_with_values = list(zip(select_new_ys, cur_values))
        ys_with_answer = [(y, v) for y, v in ys_with_values if "Answer:" in y]
        all_ys_with_answers.extend(ys_with_answer)
        ys_with_answer_sure = [(y, v) for y, v in ys_with_answer if v >= 20]
        if ys_with_answer_sure and step <= task.steps:
            y_max = max(ys_with_answer_sure, key=lambda x: x[1])[0]
            return [y_max], {'steps': infos}

        # exceed maximum number of steps
        if step > task.steps:
            if all_ys_with_answers:
                y_max = max(all_ys_with_answers, key=lambda x: x[1])[0]
                return [y_max], {'steps': infos}
            else:
                return [''], {'steps': infos}

        ys = select_new_ys
        if len(ys) < args.n_select_sample:
            ys.append('')

        step += 1


def naive_solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, args.apply_skills, args.decompose_problem, args.backend)
    return ys, {}