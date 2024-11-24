import itertools
import numpy as np
from functools import partial
from tot.models import get_output

def get_value(task, x, y, n_evaluate_sample, model, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = get_output(value_prompt, n=n_evaluate_sample, model=model, stop=None)
    value = task.value_outputs_unwrap(value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, model, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, model=model, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_proposals(task, x, y, n_generate_sample, model): 
    propose_prompt = task.propose_prompt_wrap(x, model, y)
    proposals = get_output(propose_prompt, n=n_generate_sample, model=model, stop=None)
    print(proposals)
    return proposals

def solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    values = []
    infos = []
    ids = []
    for step in range(task.steps):
        # generation
        new_ys = [get_proposals(task, x, y, args.n_generate_sample, args.backend) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        values = get_values(task, x, new_ys, args.n_evaluate_sample, args.backend, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    
    ys_with_values = list(zip(ys, final_values))
    ys_filtered = [y for y, v in ys_with_values if "Answer: " in y and v >= 20]

    if not ys_filtered:
        # if no y with "Answer: " in it has value >= 20, then take the y with "Answer: " in it with the largest value
        ys_with_answer = [(y, v) for y, v in ys_with_values if "Answer: " in y]
        if ys_with_answer:
            y_max = max(ys_with_answer, key=lambda x: x[1])[0]
            ys_filtered = [y_max]
        else:
            # If no y contains "Answer: ", take the y with the largest value
            y_max = max(ys_with_values, key=lambda x: x[1])[0]
            ys_filtered = [y_max]

    return ys_filtered, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    naive_prompt = task.naive_prompt_wrap(x, '')
    ys = get_output(naive_prompt, n=args.n_generate_sample, model=args.backend, stop=None)
    return ys, {}