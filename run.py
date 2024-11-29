import os
import json
import argparse

from tot.tasks import get_task
from tot.methods.bfs import solve, naive_solve
from tot.models import usage

def run(args):
    task = get_task(args.task)
    logs, cnt_correct = [], 0
    if args.naive_run:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_sample_{args.prompt_sample}_{args.n_generate_sample}_apply_skills_{args.apply_skills}_decompose_problem_{args.decompose_problem}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.n_generate_sample}_{args.n_evaluate_sample}_{args.method_select}_{args.n_select_sample}_apply_skills_{args.apply_skills}_decompose_problem_{args.decompose_problem}_start{args.task_start_index}_end{args.task_end_index}_retry.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i) 
        else:
            ys, info = solve(args, task, i)

        # log
        infos = [task.test_output(i, y, args.backend) for y in ys]

        # log main metric
        accs = [_['r'] for _ in infos]
        
        # log main metric
        cnt_correct += sum(accs) / len(accs)
        cur_acc = cnt_correct / (i - args.task_start_index + 1)
        print('current accuracy: ', cur_acc)
        if args.backend == 'o1-mini' or args.backend == 'gpt-4o' or args.backend == 'gpt-4o-mini':
            info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': usage(args.backend), 'current accuracy': cur_acc})
        else:
            info.update({'idx': i, 'ys': ys, 'infos': infos, 'current accuracy': cur_acc})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
    
    n = args.task_end_index - args.task_start_index
    print(cnt_correct / n)
    if args.backend == 'o1-mini' or args.backend == 'gpt-4o' or args.backend == 'gpt-4o-mini':
        print('usage_so_far', usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['o1-mini', 'gpt-4o', 'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct', 'Qwen2.5-1.5B-Instruct', 'gpt-4o-mini', 'Claude-3.5-Sonnet'], default='Claude-3.5-Sonnet')
    args.add_argument('--temperature', type=float, default=0.7) # only used for proposal; for value prompt, temperature is set as 0.1

    args.add_argument('--task', type=str, required=True, choices=['MATH', "MATH2"])
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--apply_skills', action='store_true')
    args.add_argument('--decompose_problem', action='store_true') # haven't implemented the case where apply_skills and decompose_problem are simultaneously true. only used for math^2
    args.add_argument('--n_generate_sample', type=int, default=1) 
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)