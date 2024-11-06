import itertools
import numpy as np
from functools import partial
from tot.models import get_output

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = get_output(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_proposals(task, x, y, n_generate_sample): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = get_output(propose_prompt, n=n_generate_sample, stop=None)
    print(proposals)
    return proposals

def solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        new_ys = [get_proposals(task, x, y, args.n_generate_sample) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        values = get_values(task, x, new_ys, args.n_evaluate_sample)

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
    
    # only take ys with "Answer: " in it. Is it the correct thing to do?
    ys = [y for y in ys if "Answer: " in y]
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global get_output
    get_output = partial(get_output, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_proposals(task, x, '', args.n_generate_sample, stop=None)
    return ys, {}