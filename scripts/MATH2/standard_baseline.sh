python3.11 -B run.py \
    --backend o1-mini \
    --task MATH2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 8 \
    ${@}