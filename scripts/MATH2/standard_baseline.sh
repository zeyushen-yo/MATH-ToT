python3 run.py \
    --backend gpt-4o \
    --task MATH2 \
    --task_start_index 0 \
    --task_end_index 100 \
    --naive_run \
    --prompt_sample standard \
    --n_generate_sample 20 \
    ${@}