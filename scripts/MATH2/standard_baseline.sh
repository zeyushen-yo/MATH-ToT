python3.11 -B run.py \
    --backend gpt-4o-mini \
    --task MATH2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --temperature 0 \
    --naive_run \
    --simplify_problem \
    --prompt_sample standard \
    --n_generate_sample 8 \
    ${@}